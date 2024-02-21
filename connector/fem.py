import ufl.algebra
from tqdm import tqdm
import numpy as np
import dolfin
import dolfin_adjoint
import pyadjoint.overloaded_function
from connector.sdf import backend_signed_distance_function, SignedDistanceFunctionBlock
from connector.mesh import get_mesh_mapping
from connector.shapes import get_shape, on_contact
from connector.line_fitting import backend_line_fitting, LineFittingBlock
from connector.visualization import plot_mesh
from elasticity.stress_lic import draw_lic


def l2_dis(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def function_assign(mesh, function, dof_2_vertex, cond, value):
    dof_vec = function.vector()[:]
    for vtx_idx, vtx in enumerate(mesh.coordinates()):
        if cond(vtx):
            dof_idx = dof_2_vertex[vtx_idx]
            dof_vec[dof_idx:dof_idx + 2] = value
    function.vector()[:] = dof_vec


def squared_soft_relu(x, opt):
    value = ufl.ln(1. + ufl.exp(opt.soft_relu_scale * x)) / opt.soft_relu_scale
    return value * value


class FEM:
    def __init__(self, side, shape_params, opt, print_contact_vtx=False):
        self.side = side
        self.opt = opt
        self.shape = get_shape(side=side, shape_params=shape_params, opt=opt)
        self.mesh = self.shape.mesh
        self.ds = self.shape.ds
        self.mu = opt.young_modulus / (2. * (1. + opt.poisson_ratio))
        self.lmbda = opt.young_modulus * opt.poisson_ratio / ((1. + opt.poisson_ratio) * (1. - opt.poisson_ratio))
        self.traction = dolfin_adjoint.Constant((opt.traction, 0.))
        self.pen_w = dolfin_adjoint.Constant(opt.pen_w)
        self.vfs = dolfin.VectorFunctionSpace(self.mesh, "CG", 1)
        self.tfs = dolfin.TensorFunctionSpace(self.mesh, "DG", 0)
        self.mesh_offset = dolfin_adjoint.Function(self.vfs)
        self.dof_2_vertex = get_mesh_mapping(mesh=self.mesh, function_space=self.vfs, direction='dof_2_vertex')
        if opt.enable_offset == side:
            def close_to(vtx):
                return l2_dis(vtx, opt.target_vtx) < opt.eps
            function_assign(mesh=self.mesh, function=self.mesh_offset, dof_2_vertex=self.dof_2_vertex,
                            cond=close_to, value=opt.offset)
        if opt.ctrl_var == 'mesh_offset':
            self.adjoint_ctrl = dolfin_adjoint.Control(self.mesh_offset)
        dolfin.ALE.move(self.mesh, self.mesh_offset)

        self.du = dolfin.TrialFunction(self.vfs)  # Incremental displacement
        self.v = dolfin.TestFunction(self.vfs)  # Test function
        self.u = dolfin_adjoint.Function(self.vfs)  # Displacement from previous iteration
        self.bcs = \
            [dolfin_adjoint.DirichletBC(self.vfs.sub(1), dolfin_adjoint.Constant(0.),
                                        'near(x[1], {}) && on_boundary'.format(self.shape.h / 2.), 'topological')]
        grad_u = dolfin.grad(self.u)
        strain = 0.5 * (grad_u + grad_u.T)
        self.e = self.lmbda / 2 * dolfin.tr(strain) ** 2 + self.mu * dolfin.inner(strain, strain)
        self.idt = dolfin.SpatialCoordinate(self.mesh)
        if opt.ctrl_var == 'spatial_coordinate':
            self.idt = dolfin_adjoint.project(self.idt, self.vfs)
            self.adjoint_ctrl = dolfin_adjoint.Control(self.idt)
        self.idx_lists = [[] for _ in range(len(self.shape.contact_ids))]
        for idx, vtx in enumerate(self.mesh.coordinates()):
            for contact_count, contact_id in enumerate(self.shape.contact_ids):
                if on_contact(x=vtx, x_1=self.shape.point_list[contact_id][0], y_1=self.shape.point_list[contact_id][1],
                              x_2=self.shape.point_list[contact_id + 1][0],
                              y_2=self.shape.point_list[contact_id + 1][1], incl_endpts=opt.incl_endpts, eps=opt.eps):
                    self.idx_lists[contact_count].append(self.dof_2_vertex[idx])
                    if print_contact_vtx:
                        print('[{:.15f}, {:.15f}], '.format(vtx[0], vtx[1]))
        self.idx_lists = [np.array(idx_list) // 2 for idx_list in self.idx_lists]
        if opt.penalization_type == 'line_fitting':
            self.line_fitting = pyadjoint.overloaded_function.overload_function(backend_line_fitting, LineFittingBlock)
        elif opt.penalization_type == 'sdf':
            self.consol_idx_list = []
            for item in self.idx_lists:
                self.consol_idx_list.extend(item)
            self.consol_idx_list = np.unique(self.consol_idx_list)
            self.sdf = pyadjoint.overloaded_function.overload_function(backend_signed_distance_function,
                                                                       SignedDistanceFunctionBlock)
        else:
            raise Exception

    def vis_stress(self, step_size=0.2):
        grad_u = dolfin.grad(self.u)
        strain = 0.5 * (grad_u + grad_u.T)
        epsilon = dolfin.variable(strain)
        energy = self.lmbda / 2. * dolfin.tr(epsilon) ** 2 + self.mu * dolfin.inner(epsilon, epsilon)
        sigma = dolfin.diff(energy, epsilon)
        sigma = dolfin_adjoint.project(sigma, self.tfs)

        coordinates = np.array(self.shape.mesh.coordinates())
        x_min, x_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
        y_min, y_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
        n_x = round((x_max - x_min) / step_size)
        n_y = round((y_max - y_min) / step_size)
        x_space = np.linspace(x_min, x_max, n_x)
        y_space = np.linspace(y_min, y_max, n_y)

        stress = np.zeros((n_x, n_y, 2, 2))
        mask = np.zeros((n_x, n_y), dtype=bool)
        for idx_x, x in tqdm(enumerate(x_space)):
            for idx_y, y, in enumerate(reversed(y_space)):
                answer = np.zeros(4)
                try:
                    sigma.eval(answer, x=np.array([x, y]))
                    stress[idx_x, idx_y] = answer.reshape((2, 2))
                    mask[idx_x, idx_y] = True
                except RuntimeError:
                    mask[idx_x, idx_y] = False

        draw_lic(stress=stress, mask=mask, scale=5, spacing=4)

    def solve(self, other, opt, need_disp=False, vis_mesh=False, vis_stress=False):
        E = self.e * dolfin.dx
        if self.side == 'left':
            E = E - dolfin.dot(-self.traction, self.u) * self.ds(1)
        elif self.side == 'right':
            E = E - dolfin.dot(self.traction, self.u) * self.ds(1)
        else:
            raise Exception

        if opt.penalization_type == 'line_fitting':
            # proj_u_list = []
            for contact_count, contact_id in enumerate(self.shape.contact_ids):
                line = self.line_fitting(dolfin_adjoint.project(other.u + other.idt, other.vfs),
                                         other.idx_lists[contact_count])
                f = self.idt[1] - self.idt[0] * line[1] - line[0]
                proj_u = (self.u[1] - self.u[0] * line[1] + f) / ((1. + line[1] ** 2) ** 0.5)
                on_right = 1. if self.side == 'right' else -1.
                x1_l_x2 = 1. if other.shape.point_list[contact_id][0] < other.shape.point_list[contact_id + 1][0] else -1.
                multiplier = on_right * x1_l_x2
                E = E + self.pen_w * squared_soft_relu(multiplier * proj_u, opt) * self.ds(contact_count + 2)
                # proj_u_list.append(multiplier * proj_u)
        elif opt.penalization_type == 'sdf':
            u_vec = np.reshape(self.u.vector()[:], (-1, 2))
            u_vec[self.idx_lists[0], 0] = .1
            self.u.vector()[:] = u_vec.flatten()

            func = dolfin_adjoint.project(self.idt + self.u, self.vfs)
            sdf = self.sdf(func=func, idx_list=self.consol_idx_list, mesh=self.mesh, other_mesh=other.mesh)

            tmp_func = dolfin_adjoint.Function(self.vfs)
            result_vec = np.reshape(sdf.vector()[:], (-1, 4))[:, 2:]
            tmp_func.vector()[:] = result_vec.flatten()
            import matplotlib.pyplot as plt
            plt.figure(dpi=300)
            plt.colorbar(dolfin.plot(tmp_func))
            plt.show()

            proj_u = (sdf[0] - func[0]) * sdf[2] + (sdf[1] - func[1]) * sdf[3]

            plt.figure(dpi=300)
            plt.colorbar(dolfin.plot(proj_u))
            plt.show()

            for contact_count, contact_id in enumerate(self.shape.contact_ids):
                print(dolfin.assemble(proj_u * self.ds(contact_count + 2)))
                print(dolfin.assemble(self.pen_w * squared_soft_relu(proj_u, opt) * self.ds(contact_count + 2)))
                E = E + self.pen_w * squared_soft_relu(proj_u, opt) * self.ds(contact_count + 2)

            raise NotImplementedError
        else:
            raise Exception

        dE = dolfin.derivative(E, self.u, self.v)
        jacE = dolfin.derivative(dE, self.u, self.du)

        problem = dolfin_adjoint.NonlinearVariationalProblem(dE, self.u, self.bcs, jacE)
        solver = dolfin_adjoint.NonlinearVariationalSolver(problem)
        solver.solve()
        # for idx, proj_u in enumerate(proj_u_list):
        #     print('e', float(dolfin_adjoint.assemble(E)))
        #     print('assembled', float(dolfin_adjoint.assemble(self.pen_w * soft_relu(proj_u, opt) * self.ds(idx + 2))))
        if vis_stress:
            self.vis_stress()
        if vis_mesh:
            plot_mesh((self, other), shape=self.shape, plot_flipped_mesh=True)

        if opt.disp_save_to is not None:
            self.u.rename('u', 'u')
            file = dolfin.File(opt.disp_save_to)
            file << self.u

        if need_disp:
            return dolfin_adjoint.assemble(self.u[0] * self.ds(1)) / (self.shape.h / 2.)

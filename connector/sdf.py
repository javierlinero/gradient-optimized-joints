"""
Input: Mesh
Output: Continuous Function f(pts): value (distance to mesh)
"""

import time
import numpy as np
import dolfin
import pyadjoint.overloaded_function
import matplotlib
import matplotlib.pyplot as plt
import dolfin_adjoint


class SDF:
    def __init__(self, mesh, return_gradients, plot_normal=False):
        self.mesh = mesh
        self.return_gradients = return_gradients
        self.empty_func = dolfin.Function(dolfin.FunctionSpace(self.mesh, "CG", 1))
        self.empty_answer = np.zeros(1)

        self.bmesh = dolfin_adjoint.BoundaryMesh(mesh, 'exterior')
        self.bmesh_vtx = self.bmesh.coordinates()
        self.bmesh_n_vtx = len(self.bmesh_vtx)
        self.bmesh_cells = self.bmesh.cells()
        self.bmesh_bb_tree = dolfin.BoundingBoxTree()
        self.bmesh_bb_tree.build(self.bmesh)

        self.bmesh_vtx_normals, self.bmesh_cell_normals, self.bmesh_cell_mid_points = self.get_normals()
        if plot_normal:
            self.plot_normal()

    def inside_mesh(self, point):
        try:
            self.empty_func.eval(self.empty_answer, x=np.array(point))
            return True
        except RuntimeError:
            return False

    def get_normals(self, eps=1e-5):
        bmesh_vtx_normals = [[] for _ in range(self.bmesh_n_vtx)]
        bmesh_cell_normals = []
        bmesh_cell_mid_points = []
        for idx, cell in enumerate(dolfin.cells(self.bmesh)):
            normal = np.array([cell.cell_normal()[0], cell.cell_normal()[1]])
            vtx_idx_1, vtx_idx_2 = self.bmesh_cells[idx][0], self.bmesh_cells[idx][1]
            vtx_1, vtx_2 = self.bmesh_vtx[vtx_idx_1], self.bmesh_vtx[vtx_idx_2]
            mid_point = (vtx_1 + vtx_2) / 2.
            bmesh_cell_mid_points.append(mid_point)
            plus_inside = self.inside_mesh(mid_point + eps * normal)
            minus_inside = self.inside_mesh(mid_point - eps * normal)
            assert int(plus_inside) + int(minus_inside) == 1
            if plus_inside:
                normal = normal * -1.
            bmesh_vtx_normals[vtx_idx_1].append(normal)
            bmesh_vtx_normals[vtx_idx_2].append(normal)
            bmesh_cell_normals.append(normal)
        for idx in range(self.bmesh_n_vtx):
            assert len(bmesh_vtx_normals[idx]) == 2
            bmesh_vtx_normals[idx] = bmesh_vtx_normals[idx][0] + bmesh_vtx_normals[idx][1]
            bmesh_vtx_normals[idx] = bmesh_vtx_normals[idx] / np.linalg.norm(bmesh_vtx_normals[idx])
        bmesh_vtx_normals = np.array(bmesh_vtx_normals)
        bmesh_cell_normals = np.array(bmesh_cell_normals)
        bmesh_cell_mid_points = np.array(bmesh_cell_mid_points)
        return bmesh_vtx_normals, bmesh_cell_normals, bmesh_cell_mid_points

    def plot_normal(self):
        # vertex normal
        plt.figure(dpi=300)
        plt.title('vertex normal')
        plt.gca().set_aspect('equal', 'box')
        dolfin.plot(self.mesh)
        plt.quiver(self.bmesh_vtx[:, 0], self.bmesh_vtx[:, 1],
                   self.bmesh_vtx_normals[:, 0], self.bmesh_vtx_normals[:, 1])
        plt.show()

        # cell normal
        plt.figure(dpi=300)
        plt.title('cell normal')
        plt.gca().set_aspect('equal', 'box')
        dolfin.plot(self.mesh)
        plt.quiver(self.bmesh_cell_mid_points[:, 0], self.bmesh_cell_mid_points[:, 1],
                   self.bmesh_cell_normals[:, 0], self.bmesh_cell_normals[:, 1])
        plt.show()

    def query(self, query_points, eps=1e-2, vis_grad=False):
        distances = []
        closest_points = []
        gradients = []
        for query_point in query_points:
            index, distance = self.bmesh_bb_tree.compute_closest_entity(dolfin.Point(query_point))
            inside = self.inside_mesh(query_point)
            if inside:
                distance = distance * -1.
            closest_cell = self.bmesh_cells[index]
            vertices_of_closest_cell = self.bmesh_vtx[closest_cell]
            closest_point, closest_point_status = self.get_closest_point(query_point, *vertices_of_closest_cell)
            closest_points.append(closest_point)
            direction_from_surface = query_point - closest_point
            near_surface = np.abs(distance) < eps
            if near_surface:
                if closest_point_status == 'segment':
                    gradient = self.bmesh_cell_normals[index]
                elif closest_point_status == 'a':
                    gradient = self.bmesh_vtx_normals[closest_cell[0]]
                elif closest_point_status == 'b':
                    gradient = self.bmesh_vtx_normals[closest_cell[1]]
                else:
                    raise Exception
            elif inside:
                gradient = -direction_from_surface
            else:
                gradient = direction_from_surface
            gradient = gradient / np.linalg.norm(gradient)
            gradients.append(gradient)
            distances.append(distance)
        distances = np.array(distances)
        closest_points = np.array(closest_points)
        gradients = np.array(gradients)
        if vis_grad:
            self.vis_grad(query_points, distances, gradients)
        # print(query_points)
        return closest_points, gradients

    @staticmethod
    def get_closest_point(pt, vert_a, vert_b):
        u = vert_a - vert_b
        v = pt - vert_b
        if u.dot(v) <= 0:
            return vert_b, 'b'
        if u.dot(v) >= u.dot(u):
            return vert_a, 'a'
        u_norm = u / np.linalg.norm(u)
        return u_norm.dot(v) * u_norm + vert_b, 'segment'

    def vis_grad(self, query_points, distances, gradients):
        plt.figure(dpi=300)
        plt.scatter(self.bmesh_vtx[:, 0], self.bmesh_vtx[:, 1], s=2., color='black')
        colormap = matplotlib.cm.GnBu
        norm = matplotlib.colors.Normalize()
        norm.autoscale(distances)
        sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
        plt.quiver(query_points[:, 0], query_points[:, 1], gradients[:, 0], gradients[:, 1],
                   color=colormap(norm(distances)), scale=50.)
        plt.colorbar(sm)
        plt.show()


def backend_signed_distance_function(func, idx_list, mesh, other_mesh, adj_input=None):
    print('backend_signed_distance_function')
    sdf = SDF(other_mesh, return_gradients=adj_input is not None)
    func_vector = np.reshape(func.vector()[:], (-1, 2))
    query_points_list = func_vector[idx_list]

    if adj_input is None:
        closest_points, directions = sdf.query(query_points=query_points_list)
        answer_func = dolfin_adjoint.Function(dolfin.VectorFunctionSpace(mesh, 'CG', 1, dim=4))
        vec = np.zeros((func_vector.shape[0], 4))
        # print(closest_points.shape, directions.shape)
        vec[idx_list, 0:2] = closest_points
        vec[idx_list, 2:4] = directions
        answer_func.vector()[:] = vec.flatten()
        return answer_func
        # closest_point_func = dolfin_adjoint.Function(func.function_space())
        # vec = np.zeros(func_vector)
        # vec[idx_list] = closest_points
        # closest_point_func.vector()[:] = vec.flatten()
        #
        # direction_func = dolfin_adjoint.Function(func.function_space())
        # vec = np.zeros(func_vector)
        # vec[idx_list] = directions
        # direction_func.vector()[:] = vec.flatten()
    else:
        raise NotImplementedError
        # adj_inputs_vec = np.reshape(adj_input.vec()[:], (-1, 2))
        # ans_func = dolfin_adjoint.Function(vfs)
        # vec = np.zeros_like(func_vector)
        # vec[idx_list] = answer[:, np.newaxis] * adj_inputs_vec
        # ans_func.vector()[:] = vec.flatten()
        # return ans_func


class SignedDistanceFunctionBlock(pyadjoint.Block):
    def __init__(self, func, idx_list, mesh, other_mesh, **kwargs):
        super(SignedDistanceFunctionBlock, self).__init__()
        self.idx_list = idx_list
        self.mesh = mesh
        self.other_mesh = other_mesh
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'SignedDistanceFunctionBlock'

    def recompute_component(self, inputs, block_variable, idx, prepared):
        print('recompute_component')
        assert len(inputs) == 1 and idx == 0
        return backend_signed_distance_function(inputs[0], self.idx_list, self.mesh, self.other_mesh)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        print('evaluate_adj_component')
        assert len(inputs) == 1 and len(adj_inputs) == 1 and idx == 0
        return backend_signed_distance_function(inputs[0], self.idx_list, self.other_mesh, self.mesh,
                                                adj_input=adj_inputs[0])


if __name__ == '__main__':
    sdf = pyadjoint.overloaded_function.overload_function(backend_signed_distance_function, SignedDistanceFunctionBlock)
    from args import parse_args
    from shapes import get_shape
    from fem import FEM
    opt = parse_args()
    left = FEM(side='left', shape_params=opt.init_shape_params, opt=opt)
    right = FEM(side='right', shape_params=opt.init_shape_params, opt=opt)
    left_mesh = left.mesh
    right_mesh = right.mesh
    sfs = dolfin.FunctionSpace(left_mesh, 'CG', 1)
    vfs = dolfin.VectorFunctionSpace(left_mesh, 'CG', 1)
    offset = dolfin_adjoint.Expression(("0.", "0."), element=vfs.ufl_element())
    func = dolfin_adjoint.project(dolfin.SpatialCoordinate(left_mesh), vfs)
    index_list = []
    for item in left.idx_lists:
        index_list.extend(item)
    index_list = np.unique(index_list)
    # print(len(index_list), index_list)
    ans_func = sdf(func=func, idx_list=index_list, mesh=left_mesh, other_mesh=right_mesh)
    func = dolfin_adjoint.Function(vfs)
    result_vec = np.reshape(ans_func.vector()[:], (-1, 4))[:, :2]
    print(result_vec)
    func.vector()[:] = result_vec.flatten()

    print("Answer vec:")
    ans_vec = ans_func.vector()[:][index_list]
    print(ans_vec)
    print(np.min(ans_vec), np.max(ans_vec))
    plt.figure(dpi=300)
    plt.colorbar(dolfin.plot(func, scale=500.))
    dolfin.plot(right_mesh)
    plt.show()
    #ans_vec = ans_func.vector()[:][index_list]
    #print(ans_vec)
    #print(np.min(ans_vec), np.max(ans_vec))

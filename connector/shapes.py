import numpy as np
import torch
import dolfin
import meshio
import pygmsh
import tempfile
import matplotlib.pyplot as plt
import dolfin_adjoint
from visualization import dolfin_plot, plot_joint


def plot_line_segments(point_lists):
    plt.figure(dpi=200)
    plt.gca().set_aspect('equal', 'box')
    for point_list in point_lists:
        point_list = np.concatenate([point_list, point_list[0:1]])
        plt.plot(point_list[:, 0], point_list[:, 1], color='black')
    plt.show()


def two_d_tensor(a, b):
    if isinstance(a, float):
        a = torch.tensor(a, dtype=torch.float64)
    if isinstance(b, float):
        b = torch.tensor(b, dtype=torch.float64)
    return torch.cat([a.unsqueeze(dim=0), b.unsqueeze(dim=0)], dim=0).unsqueeze(dim=0)


def on_contact(x, x_1, y_1, x_2, y_2, incl_endpts, eps):
    len_01 = np.linalg.norm([x_1 - x[0], y_1 - x[1]])
    len_02 = np.linalg.norm([x_2 - x[0], y_2 - x[1]])
    if not incl_endpts and (len_01 < eps or len_02 < eps):
        return False

    lhs = (x[1] - y_1) * (x_1 - x_2)
    rhs = (x[0] - x_1) * (y_1 - y_2)
    return min(x_1, x_2) - eps <= x[0] <= max(x_1, x_2) + eps and \
        min(y_1, y_2) - eps <= x[1] <= max(y_1, y_2) + eps and lhs - eps <= rhs <= lhs + eps


class BaseShape:
    def __init__(self, side, shape_params, opt):
        self.side = side
        self.shape_params = shape_params
        self.opt = opt
        self.point_list, self.shape_params_tensor, self.point_list_tensor = self.get_point_list()
        # plot_line_segments([self.point_list])
        self.mesh, self.ds = self.create_mesh()

    def get_point_list(self):
        raise NotImplementedError

    def create_mesh(self):
        with tempfile.NamedTemporaryFile(suffix='.xml') as xml_file:
            with pygmsh.occ.Geometry() as geom:
                geom.add_polygon(self.point_list, mesh_size=self.opt.mesh_size)
                msh = geom.generate_mesh()
                triangle_mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": msh.cells[1].data})
                triangle_mesh.write(xml_file.name)
            mesh = dolfin_adjoint.Mesh(xml_file.name)
        boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        boundary_y = 0. if self.side == 'left' else self.w

        class Boundary(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return dolfin.near(x[0], boundary_y) and on_boundary

        Boundary().mark(boundaries, 1)
        for count, contact_id in enumerate(self.contact_ids):
            class Contact(dolfin.SubDomain):
                def inside(domain_self, x, on_boundary):
                    return on_contact(x=x, x_1=self.point_list[contact_id][0], y_1=self.point_list[contact_id][1],
                                      x_2=self.point_list[contact_id + 1][0], y_2=self.point_list[contact_id + 1][1],
                                      incl_endpts=self.opt.incl_endpts, eps=self.opt.eps) and on_boundary

            Contact().mark(boundaries, count + 2)
        ds = dolfin.Measure('ds', subdomain_data=boundaries)
        # for idx in range(len(self.contact_ids) + 2):
        #     print(idx, ds.subdomain_data().where_equal(idx))
        return mesh, ds

    def visualize(self):
        dolfin_plot(self.mesh, no_axis=True)


class SimpleJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 25.
        self.contact_ids = [2]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        assert len(self.shape_params) == 3
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w / 2., 0.),
             two_d_tensor(self.w / 2., shape_params_tensor[0]),
             two_d_tensor(self.w / 2. + shape_params_tensor[1], shape_params_tensor[2]),
             two_d_tensor(self.w / 2. + shape_params_tensor[1], self.h / 2.)]
        x = 0. if self.side == 'left' else self.w
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)

        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor


class SingleJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 25.
        self.contact_ids = [2, 3, 4]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        assert len(self.shape_params) == 6
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w / 2., 0.),
             two_d_tensor(self.w / 2., shape_params_tensor[1]),
             two_d_tensor(self.w / 2. + shape_params_tensor[0], shape_params_tensor[1]),
             two_d_tensor(self.w / 2. + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w / 2. + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w / 2. + shape_params_tensor[4], self.h / 2.)]
        x = 0. if self.side == 'left' else self.w
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)

        # grad test
        # shape_params_tensor.retain_grad()
        # point_list_tensor.backward(gradient=torch.tensor([[0., 0.], [0., 0.], [0., 0.], [0., 0.],
        #                                                   [0., 0.], [0., 0.], [0., 0.], [0., 1.]]))
        # print(shape_params_tensor.grad.detach().cpu().numpy())

        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor


class DoubleJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 60.
        self.w2 = 40.
        self.h = 40.
        self.h2 = 25.
        self.theta = np.pi / 10.
        self.sin = np.sin(self.theta)
        self.cos = np.cos(self.theta)
        self.tan = np.tan(self.theta)
        self.contact_ids = [2, 3, 5]
        self.traction_len = self.h2
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        assert len(self.shape_params) == 6
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        # theta = shape_params_tensor[0]
        # sin = torch.sin(theta)
        # cos = torch.cos(theta)
        # tan = torch.tan(theta)
        point_list = \
            [two_d_tensor(self.w / 2., 0.),
             two_d_tensor(self.w / 2. + shape_params_tensor[0] * self.sin, shape_params_tensor[0] * self.cos),
             two_d_tensor(self.w / 2. + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w / 2. + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w / 2. + shape_params_tensor[1] * self.sin, shape_params_tensor[1] * self.cos),
             two_d_tensor(self.w / 2. + self.h / 2. * self.tan, self.h / 2.)]
        x1 = 0. if self.side == 'left' else self.w
        x2 = (self.w - self.w2) / 2. if self.side == 'left' else (self.w + self.w2) / 2.
        point_list = [two_d_tensor(x1, (self.h - self.h2) / 2.), two_d_tensor(x2, (self.h - self.h2) / 2.)] + point_list
        point_list = point_list + [two_d_tensor(x2, self.h / 2.), two_d_tensor(x1, self.h / 2.)]

        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

# adding new joints here
class GooseNeckJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 50.
        self.h = 25.
        self.contact_ids = [1] # update contact ids later
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)
    
    def get_point_list(self):
        assert len(self.shape_params) == 6
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w * 0.4, 0.),
             two_d_tensor(self.w * 0.4, shape_params_tensor[0]),
             two_d_tensor(self.w * 0.4 + shape_params_tensor[1], shape_params_tensor[0]),
             two_d_tensor(self.w * 0.4 + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w * 0.4 + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w * 0.4 + shape_params_tensor[4], self.h / 2)
             ]
        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class ScarfJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 50.
        self.contact_ids = [1]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        assert len(self.shape_params) == 6
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w / 8., 0.),
             two_d_tensor(self.w / 8. + shape_params_tensor[0], shape_params_tensor[1]),
             two_d_tensor(self.w / 8. + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w / 8. + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w / 8. + shape_params_tensor[4] + shape_params_tensor[2] - shape_params_tensor[0], shape_params_tensor[5] + shape_params_tensor[3] - shape_params_tensor[1]),
             two_d_tensor(self.w - self.w / 8., self.h / 2.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class LapJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 50.
        self.contact_ids = [1]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        assert len(self.shape_params) == 6
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w / 4., 0.),
             two_d_tensor(self.w / 4., shape_params_tensor[0]),
             two_d_tensor(self.w / 4. + shape_params_tensor[1], shape_params_tensor[0]),
             two_d_tensor(self.w / 4. + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w / 4. + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w  - self.w / 4., self.h/2)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class DoveScarfJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 50.
        self.contact_ids = [1]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)
    
    def get_point_list(self):
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w / 8. + shape_params_tensor[0], 0.),
             two_d_tensor(self.w / 8. + shape_params_tensor[0], shape_params_tensor[1]),
             two_d_tensor(self.w / 8., shape_params_tensor[2]),
             two_d_tensor(self.w / 8., shape_params_tensor[3]),
             two_d_tensor(self.w / 8. + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w / 8 + shape_params_tensor[6], shape_params_tensor[7]),
             two_d_tensor(self.w - self.w / 8., self.h / 2. - shape_params_tensor[3]),
             two_d_tensor(self.w - self.w / 8., self.h / 2. - shape_params_tensor[2]),
             two_d_tensor(self.w - self.w / 8. - shape_params_tensor[0], self.h / 2. - shape_params_tensor[1]),
             two_d_tensor(self.w - self.w / 8. - shape_params_tensor[0], self.h / 2.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class RabbetJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 50.
        self.contact_ids = [1]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w / 4., 0.),
             two_d_tensor(self.w / 4., shape_params_tensor[0]),
             two_d_tensor(self.w / 4. + shape_params_tensor[1], shape_params_tensor[0]),
             two_d_tensor(self.w / 4. + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w / 4. + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w / 4. + shape_params_tensor[4], shape_params_tensor[0]),
             two_d_tensor(self.w / 4. + shape_params_tensor[6], shape_params_tensor[7]),
             two_d_tensor(self.w / 4. + shape_params_tensor[6], self.h / 2.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class TestJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 60.
        self.h = 70.
        self.contact_ids = [1]
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)

    def get_point_list(self):
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(shape_params_tensor[0], 0.),
             two_d_tensor(shape_params_tensor[0], 5.),
             two_d_tensor(15., 10.),
             two_d_tensor(15., 15.),
             two_d_tensor(10., 15.),
             two_d_tensor(10., 20.),
             two_d_tensor(15., 20.),
             two_d_tensor(15., 25.),
             two_d_tensor(20., 30.),
             two_d_tensor(25., 30.),
             two_d_tensor(25., 15.),
             two_d_tensor(30., 25.),
             two_d_tensor(30., 30.),
             two_d_tensor(45., 30.),
             two_d_tensor(45., 35.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor


def get_shape(side, shape_params, opt):
    if opt.shape_name == 'simple_joint':
        shape_class = SimpleJoint
    elif opt.shape_name == 'single_joint':
        shape_class = SingleJoint
    elif opt.shape_name == 'double_joint':
        shape_class = DoubleJoint
    elif opt.shape_name == 'gooseneck_joint':
        shape_class = GooseNeckJoint
    elif opt.shape_name == 'scarf_joint':
        shape_class = ScarfJoint
    elif opt.shape_name == 'lap_joint':
        shape_class = LapJoint
    elif opt.shape_name == 'dovetail_scarf_joint':
        shape_class = DoveScarfJoint
    elif opt.shape_name == 'rabbet_joint':
        shape_class = RabbetJoint
    elif opt.shape_name == 'test_joint':
        shape_class = TestJoint
    else:
        raise Exception
    return shape_class(side=side, shape_params=shape_params, opt=opt)


if __name__ == '__main__':
    from args import parse_args
    opt = parse_args()
    # if none defined simple joint
    #opt.shape_name = 'gooseneck_joint'
    #opt.init_shape_params = [6.25, 5., 6.5, 3.5, 16, 7.5]
    
    #opt.shape_name = 'scarf_joint'
    #opt.init_shape_params = [1.5, 5., 14.5, 14.5, 13.5, 10.5]

    #opt.shape_name = 'lap_joint'
    #opt.init_shape_params = [15., 5., 5., 10., 20., 10.]

    #opt.shape_name = 'dovetail_scarf_joint'
    #opt.init_shape_params = [2.5, 7.5, 5., 15., 14.5, 11.5, 12.5, 14.5]

    #opt.shape_name = 'double_joint'
    #opt.init_shape_params = [10., 14., 4., 6., 10., 12.]

    #opt.shape_name = 'rabbet_joint'
    #opt.init_shape_params = [20., 10., 10., 10., 15., 10., 25., 20.]

    opt.shape_name = 'test_joint'
    opt.init_shape_params = [10.]

    l = get_shape(side='left', shape_params=opt.init_shape_params, opt=opt)
    r = get_shape(side='right', shape_params=opt.init_shape_params, opt=opt)
    plot_joint(l, r, 40.)
    #dolfin_plot(l.mesh)
    #dolfin_plot(r.mesh)

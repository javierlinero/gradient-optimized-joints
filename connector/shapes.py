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
        self.contact_ids = [2, 3, 4] # update contact ids later
        self.traction_len = self.h
        super().__init__(side, shape_params, opt)
    
    def get_point_list(self):
        assert len(self.shape_params) == 6
        shape_params_tensor = torch.tensor(self.shape_params, requires_grad=True, dtype=torch.float64)
        point_list = \
            [two_d_tensor(self.w * 0.3, 0.),
             two_d_tensor(self.w * 0.3, shape_params_tensor[0]),
             two_d_tensor(self.w * 0.3 + shape_params_tensor[1], shape_params_tensor[0]),
             two_d_tensor(self.w * 0.3 + shape_params_tensor[2], shape_params_tensor[3]),
             two_d_tensor(self.w * 0.3 + shape_params_tensor[4], shape_params_tensor[5]),
             two_d_tensor(self.w * 0.3 + shape_params_tensor[4], self.h / 2)
             ]
        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class ScarfJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 25.
        self.contact_ids = [2,3,4]
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
             two_d_tensor(self.w - self.w / 8. - shape_params_tensor[0], self.h / 2. - shape_params_tensor[1]),
             two_d_tensor(self.w - self.w / 8., self.h / 2.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class LapJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 25.
        self.h = 25.
        self.contact_ids = [2,3,4]
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
             two_d_tensor(self.w / 4. + shape_params_tensor[4], self.h / 2.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class DoveScarfJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 40.
        self.h = 25.
        self.contact_ids = [2,4,5,8]
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
             two_d_tensor(self.w / 8. + shape_params_tensor[6] + shape_params_tensor[4], self.h / 2. - shape_params_tensor[3]),
             two_d_tensor(self.w / 8. + shape_params_tensor[6] + shape_params_tensor[4], self.h / 2. - shape_params_tensor[2]),
             two_d_tensor(self.w / 8. + shape_params_tensor[6] + shape_params_tensor[4] - shape_params_tensor[0], self.h / 2. - shape_params_tensor[1]),
             two_d_tensor(self.w / 8. + shape_params_tensor[6] + shape_params_tensor[4] - shape_params_tensor[0], self.h / 2.)
             ]

        x = 0. if self.side == 'left' else self.w 
        point_list = [two_d_tensor(x, 0.)] + point_list + [two_d_tensor(x, self.h / 2.)]
        point_list_tensor = torch.cat(point_list, dim=0)
        return point_list_tensor.cpu().detach().numpy(), shape_params_tensor, point_list_tensor

class RabbetJoint(BaseShape):
    def __init__(self, side, shape_params, opt):
        self.w = 25.
        self.h = 25.
        self.contact_ids = [2,3,4]
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

    #opt.shape_name = 'simple_joint'
    #opt.init_shape_params = [11.09522583715706, 3.402814262541338, 8.255697038658727]
    #opt.init_shape_params = [10.75663947644681, 3.2912300285786045, 7.8443222318319865]

    opt.shape_name = 'gooseneck_joint'
    opt.init_shape_params =  [5.5, 10, 11.5, 3, 20, 8.5]
    #opt.init_shape_params = 5.533996076603684, 10.30894125218098, 10.499117006411245, 3.064340846513361, 20.395316162803233, 8.597639121512778
    #opt.init_shape_params = 5.806438738845788, 5.274990913767001, 5.422073628194587, 3.339200801944026, 15.217181433445996, 8.611239648331097
    #opt.init_shape_params = [6.223862526151819, 4.610382705762201, 4.711375807427571, 3.724588694467258, 14.755971924872089, 8.558012231610377]
    #opt.init_shape_params = [5.605037407272536, 5.506728046189577, 5.604432227062989, 3.7894892923561656, 14.805797329280153, 8.689959557937993]
    #opt.init_shape_params =  [5.55519800090113, 5.513620885885361, 5.5469095174449645, 4.085337787107506, 14.823502449810613, 8.711441660281004]
    #opt.init_shape_params = [5.607203643794017, 5.5113608130272285, 5.558971040488088, 4.118777245861947, 15.040717304917573, 8.75985281528125]
    #opt.init_shape_params = [5.829022066726009, 5.405563486754575, 5.584668539017124, 3.3412057791790226, 14.844819582417419, 8.580114263017458]
    #opt.init_shape_params = [5.906454392706691, 5.445595364605609, 5.610540276963511, 3.417307568273767, 14.855623681506957, 8.656437612525608]

    #opt.shape_name = 'scarf_joint'
    #opt.init_shape_params = [1.5, 2.5, 14.5, 7.25, 14.5, 5.25]
    #opt.init_shape_params = [1.5437586875276876, 2.3518654593507975, 14.22267391080198, 7.705935755965082, 13.792762472092132, 5.271502102066418]
    #opt.init_shape_params = [1.5631212863541564, 2.2821623300421705, 14.297202872860344, 7.614318498916319, 13.772197633310087, 5.176078623790447]
    #opt.init_shape_params = [1.585672495044579, 2.1974015064156647, 14.305387297652876, 7.626682100834308, 13.78299262416032, 5.217622063563789]
    #opt.init_shape_params =  [1.5677684468118256, 2.019474191625762, 14.285972352448487, 7.613917249947341, 13.774640181509218, 5.177353563532924]
    #opt.init_shape_params =  [1.6434254015652183, 2.0687839880754018, 14.27036058922764, 7.69823667518269, 13.813688575301642, 5.244746689588424]
    
    #opt.shape_name = 'lap_joint'
    #opt.init_shape_params = [10., 5., 2.5, 5., 12.5, 5.]
    #opt.init_shape_params = [7.495461181692683, 4.686947340490695, 3.3663563123601126, 5.367144637753482, 12.143113180578148, 6.723346366156181]
    #opt.init_shape_params = [10., 5., 3.5, 5., 12.5, 5.]
    #opt.init_shape_params =[7.92789339340289, 4.598690097990076, 4.148167957757523, 5.504205827589188, 12.204302849772304, 7.61144267601873]

    # opt.shape_name = 'dovetail_scarf_joint'
    # opt.init_shape_params = [2.5, 3.75, 2.5, 7.5, 12.5, 7.25, 12.5, 5.75]
    #opt.init_shape_params = [2.11449452730177, 4.872657294659804, 3.4775853356072517, 6.106506766372234, 13.363328637761528, 7.6578329110580174, 13.657172140397298, 5.068218910322886]
    #opt.init_shape_params = [1.8755375735700293, 5.100059521423366, 3.464784612377812, 6.538116192946263, 13.442661315978116, 7.643346333300166, 13.577972870541789, 5.151034164939281]
    #opt.init_shape_params = [1.9426078630696757, 5.036515551287437, 3.462882733293337, 6.520557866847855, 13.206030456768593, 7.622348672672017, 13.47927540081169, 5.127610913819341]
    #opt.init_shape_params = [1.9320147321359322, 5.301400441300441, 3.3555626118017505, 7.011062523976736, 13.416264152987535, 7.669696813284092, 13.40810110088808, 5.1583967951437755]
    #opt.init_shape_params = [1.3692232390080425, 5.575476884618501, 3.4125486072798243, 7.649465261732008, 13.651587542285199, 7.774888852140875, 13.40732177025047, 5.285473330858033]
    # opt.init_shape_params = [
    #         1.3725191280063451,
    #         5.478772288080844,
    #         3.4025675136133455,
    #         7.292475084512462,
    #         12.25362789839277,
    #         8.870353082018806,
    #         12.839565718467112,
    #         3.900203156518426
    #     ]
    #opt.shape_name = 'double_joint'
    #opt.init_shape_params = [10., 14., 4., 6., 10., 12.]

    #opt.shape_name = 'rabbet_joint'
    #opt.init_shape_params = [10., 5., 5., 5., 7.5, 5., 12.5, 10.]
    #opt.init_shape_params = [8.07988537786776, 4.599768192255249, 4.6230095719768665, 5.607261511262655, 8.292233966851862, 6.284770773101101, 12.424763473308, 10.048569736831578]

    #opt.shape_name = 'test_joint'
    #opt.init_shape_params = [10.]

    l = get_shape(side='left', shape_params=opt.init_shape_params, opt=opt)
    r = get_shape(side='right', shape_params=opt.init_shape_params, opt=opt)
    #plot_joint(l, r, 40.)
    dolfin_plot(l.mesh)
    dolfin_plot(r.mesh)

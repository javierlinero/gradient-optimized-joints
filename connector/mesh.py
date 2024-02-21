import numpy as np
import dolfin
import meshio
import pygmsh
import tempfile
import dolfin_adjoint
from connector.visualization import dolfin_plot


def on_contact(x, opt):
    x_1, y_1 = opt.control_points[opt.contact_idx]
    x_2, y_2 = opt.control_points[opt.contact_idx + 1]
    x_1, x_2 = x_1 + opt.w / 2., x_2 + opt.w / 2.
    lhs = (x[1] - y_1) * (x_1 - x_2)
    rhs = (x[0] - x_1) * (y_1 - y_2)
    return min(x_1, x_2) - opt.eps <= x[0] <= max(x_1, x_2) + opt.eps and \
        min(y_1, y_2) - opt.eps <= x[1] <= max(y_1, y_2) + opt.eps and lhs - opt.eps <= rhs <= lhs + opt.eps


def get_mesh_mapping(mesh, function_space, direction):
    dof_map = function_space.dofmap()
    n_vertices = mesh.ufl_cell().num_vertices()
    indices = [dof_map.tabulate_entity_dofs(0, i)[0] for i in range(n_vertices)]
    mesh_mapping = dict()
    for cell in dolfin.cells(mesh):
        if direction == 'dof_2_vertex':
            mesh_mapping.update(dict(vd for vd in zip(cell.entities(0), dof_map.cell_dofs(cell.index())[indices])))
        elif direction == 'vertex_2_dof':
            mesh_mapping.update(dict(vd for vd in zip(dof_map.cell_dofs(cell.index())[indices], cell.entities(0))))
        else:
            raise Exception
    mesh_mapping = np.array([int(mesh_mapping[key]) for key in sorted(mesh_mapping)])
    return mesh_mapping


def get_mesh(side, opt, vis=False):
    class LeftBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], 0.) and on_boundary

    class RightBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], opt.w) and on_boundary

    class Contact(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_contact(x, opt) and on_boundary

    if side == 'left':
        point_list = [[0., opt.h / 2.], [0., 0.]]
    elif side == 'right':
        point_list = [[opt.w, opt.h / 2.], [opt.w, 0.]]
    else:
        raise Exception
    point_list.extend([[opt.w / 2., 0.], [opt.w / 2., opt.control_points[0][1]]])
    for control_point in opt.control_points:
        point_list.append([opt.w / 2. + control_point[0], control_point[1]])
    point_list.append([opt.w / 2. + opt.control_points[-1][0], opt.h / 2.])
    opt.edge_list = point_list[len(point_list) - len(opt.control_points) - 2:]
    with tempfile.NamedTemporaryFile(suffix='.xml') as xml_file:
        with pygmsh.occ.Geometry() as geom:
            geom.add_polygon(point_list, mesh_size=opt.mesh_size)
            msh = geom.generate_mesh()
            triangle_mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": msh.cells[1].data})
            triangle_mesh.write(xml_file.name)
        mesh = dolfin_adjoint.Mesh(xml_file.name)
    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    LeftBoundary().mark(boundaries, 1)
    RightBoundary().mark(boundaries, 2)
    Contact().mark(boundaries, 3)
    ds = dolfin.Measure('ds', subdomain_data=boundaries)

    if vis:
        dolfin_plot(mesh)
    return mesh, ds


if __name__ == '__main__':
    from connector.args import parse_args
    opt = parse_args()
    get_mesh(side='left', opt=opt, vis=True)
    get_mesh(side='right', opt=opt, vis=True)

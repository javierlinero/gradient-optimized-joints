import numpy as np
import dolfin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def dolfin_plot(func, scale=None, need_colorbar=False, dpi=200, title=None, xlim=None, ylim=None, no_axis=False):
    plt.figure(dpi=dpi)
    if title is not None:
        plt.title(title)
    plt.gca().set_aspect('equal', 'box')
    if scale is not None:
        colorbar = dolfin.plot(func, scale=scale)
    else:
        colorbar = dolfin.plot(func)
    if need_colorbar:
        plt.colorbar(colorbar)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if no_axis:
        plt.axis('off')
    plt.show()


def plot_mesh(fems, shape, dpi=300, plot_flipped_mesh=False):
    plt.figure(dpi=dpi)
    plt.gca().set_aspect('equal', 'box')
    for fem in fems:
        disp = np.concatenate(([fem.u.vector()[:][fem.dof_2_vertex]],
                               [fem.u.vector()[:][fem.dof_2_vertex + 1]]), axis=0).T
        coordinates = fem.mesh.coordinates() + disp
        cells = fem.mesh.cells()
        for idx_a, idx_b in [(0, 1), (1, 2), (2, 0)]:
            segment_list = [[coordinates[cells[idx][idx_a]], coordinates[cells[idx][idx_b]]]
                            for idx in range(len(cells))]
            plt.gca().add_collection(LineCollection(segment_list, colors='black', linewidths=.2))
            if plot_flipped_mesh:
                segment_list = np.array(segment_list)
                segment_list[:, :, 1] = shape.h - segment_list[:, :, 1]
                plt.gca().add_collection(LineCollection(segment_list, colors='black', linewidths=.2))
    plt.xlim(-2., shape.w + 2.)
    if plot_flipped_mesh:
        plt.ylim(-2., shape.h + 2.)
    else:
        plt.ylim(-2., shape.h / 2. + 2.)
    plt.axis('off')
    plt.show()


def plot_grad(grad, opt, dpi=200):
    if len(grad.shape) == 1:
        grad = np.reshape(grad, (-1, 2))
    plt.figure(dpi=dpi)
    plt.gca().set_aspect('equal', 'box')
    assert len(opt.control_points) == len(grad)
    for idx in range(len(opt.control_points)):
        control_point = opt.control_points[idx]
        vtx_grad = grad[idx]
        plt.arrow(control_point[0] + opt.w / 2., control_point[1], vtx_grad[0], vtx_grad[1], width=.01, color='black')
    edge_x, edge_y = list(zip(*opt.edge_list))
    plt.plot(edge_x, edge_y, color='black')
    plt.show()


def plot_joint(left, right, x_offset=7.):
    plt.figure(dpi=500)
    plt.gca().set_aspect('equal', adjustable='box')
    if left is not None:
        plt.fill(left.point_list[:, 0], left.point_list[:, 1], color='0.85')
        plt.fill(left.point_list[:, 0], left.h - left.point_list[:, 1], color='0.85')
    if right is not None:
        plt.fill(right.point_list[:, 0] + x_offset, right.point_list[:, 1], color='0.85')
        plt.fill(right.point_list[:, 0] + x_offset, right.h - right.point_list[:, 1], color='0.85')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    from connector.args import parse_args
    from connector.shapes import get_shape
    opt = parse_args()
    x0 = np.array(opt.init_shape_params)
    left_shape = get_shape(side='left', shape_params=x0, opt=opt)
    right_shape = get_shape(side='right', shape_params=x0, opt=opt)
    plot_joint(left=left_shape, right=right_shape)
    # plot_joint(left=left_shape, right=None)

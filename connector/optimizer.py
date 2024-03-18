from tqdm import tqdm
import scipy.optimize
import numpy as np
import torch
import dolfin
import functools
import matplotlib.pyplot as plt
import dolfin_adjoint
from fem import FEM
from args import parse_args
from mesh import get_mesh
from shapes import get_shape
from visualization import dolfin_plot, plot_joint, plot_grad


def on_edge(x0, y0, x1, y1, x2, y2, eps, incl_endpts):
    if not (min(x1, x2) - eps <= x0 <= max(x1, x2) + eps) or not (min(y1, y2) - eps <= y0 <= max(y1, y2) + eps):
        return None
    lhs = (y0 - y1) * (x1 - x2)
    rhs = (x0 - x1) * (y1 - y2)
    if not (lhs - eps <= rhs <= lhs + eps):
        return None
    len_12 = np.linalg.norm([x2 - x1, y2 - y1])
    len_01 = np.linalg.norm([x1 - x0, y1 - y0])
    len_02 = np.linalg.norm([x2 - x0, y2 - y0])
    if not incl_endpts and (len_01 < eps or len_02 < eps):
        return None
    return len_01 / len_12, len_02 / len_12


def two_d_tensor(a, b):
    return torch.cat([a.unsqueeze(dim=0), b.unsqueeze(dim=0)], dim=0).unsqueeze(dim=0)


def func(shape_params, ret_value, lookup, opt, vis_grad=False, vis_mesh=False):
    # print(shape_params, ret_value)
    assert ret_value in ['func', 'disp', 'grad']

    tuple_shape_params = tuple(shape_params)
    if tuple_shape_params in lookup:
        return lookup[tuple_shape_params][ret_value]

    left = FEM(side='left', shape_params=shape_params, opt=opt)
    right = FEM(side='right', shape_params=shape_params, opt=opt)

    left_disp, right_disp = None, None
    for it in range(opt.num_it):
        if it < opt.num_it - 1:
            left.solve(right, opt=opt)
            right.solve(left, opt=opt)
        else:
            left_disp = left.solve(right, need_disp=True, opt=opt)
            right_disp = right.solve(left, need_disp=True, opt=opt, vis_mesh=vis_mesh)

    # grad calculation
    disp = right_disp - left_disp
    left_grad = dolfin_adjoint.compute_gradient(disp, left.adjoint_ctrl)
    right_grad = dolfin_adjoint.compute_gradient(disp, right.adjoint_ctrl)
    vtx_grad = np.zeros(2)
    shape_params_grad = None

    regularizer = None
    for contact_id in left.shape.contact_ids:
        length = torch.sum(
            (left.shape.point_list_tensor[contact_id + 1] - left.shape.point_list_tensor[contact_id]) ** 2.) ** 0.5
        penalty = torch.clamp(opt.reg_soft_min_len - length, min=0.) ** 2
        if regularizer is None:
            regularizer = penalty
        else:
            regularizer = regularizer + penalty

    if opt.shape_name == 'simple_joint':
        width = left.shape.h - left.shape.shape_params_tensor[0] * 2.
    elif opt.shape_name == 'single_joint':
        width = left.shape.h - left.shape.shape_params_tensor[1] * 2.
    elif opt.shape_name == 'double_joint':
        width = (left.shape.shape_params_tensor[1] - left.shape.shape_params_tensor[0]) * 2.
    elif opt.shape_name == 'gooseneck_joint': # must fill in these after
        width = left.shape.h
    elif opt.shape_name == 'scarf_joint':
        width = left.shape.h
    elif opt.shape_name == 'lap_joint':
        width = left.shape.h 
    elif opt.shape_name == 'dovetail_scarf_joint':
        width = left.shape.h 
    elif opt.shape_name == 'rabbet_joint':
        width = left.shape.h 
    else:
        raise Exception
    regularizer = regularizer + torch.clamp(opt.reg_soft_min_width - width, min=0.) ** 2

    regularizer = regularizer * opt.reg_w

    for fem, grad in [(left, left_grad), (right, right_grad)]:
        vtx_tensor_list = []
        vtx_grad_list = []
        mesh = fem.mesh
        shape_params_tensor = fem.shape.shape_params_tensor
        point_list_tensor = fem.shape.point_list_tensor
        for vertex in mesh.coordinates():
            x0, y0 = vertex
            for edge_idx in range(1, len(fem.shape.point_list) - 2):
                x1, y1 = fem.shape.point_list[edge_idx]
                x2, y2 = fem.shape.point_list[edge_idx + 1]
                para = on_edge(x0=x0, y0=y0, x1=x1, y1=y1, x2=x2, y2=y2, eps=opt.eps, incl_endpts=opt.incl_endpts)
                if para is not None:
                    alpha, beta = para
                    grad.eval(vtx_grad, x=vertex)
                    vtx_tensor = point_list_tensor[edge_idx] * beta + point_list_tensor[edge_idx + 1] * alpha
                    vtx_tensor_list.append(vtx_tensor.unsqueeze(dim=0))
                    vtx_grad_list.append(np.array(vtx_grad))
                    break
        vtx_tensor = torch.cat(vtx_tensor_list, dim=0)
        shape_params_tensor.retain_grad()

        if id(fem) == id(left):
            vtx_tensor.backward(gradient=torch.tensor(vtx_grad_list), retain_graph=True)
            regularizer.backward()
        else:
            vtx_tensor.backward(gradient=torch.tensor(vtx_grad_list))

        if shape_params_grad is None:
            shape_params_grad = shape_params_tensor.grad.detach().cpu().numpy()
        else:
            shape_params_grad = shape_params_grad + shape_params_tensor.grad.detach().cpu().numpy()

    if vis_grad:
        dolfin_plot(left_grad, scale=.25)
        dolfin_plot(right_grad, scale=.25)

    # save results
    lookup[tuple_shape_params] = dict()
    lookup[tuple_shape_params]['disp'] = float(right_disp) - float(left_disp)
    lookup[tuple_shape_params]['func'] = float(right_disp) - float(left_disp) + float(regularizer)
    lookup[tuple_shape_params]['grad'] = shape_params_grad
    return lookup[tuple_shape_params][ret_value]


def random_ptb(f, shape_params, opt):
    hash_value = hash(tuple(shape_params))
    mod = 2 ** 32
    seed_value = (hash_value % mod + mod) % mod
    np.random.seed(seed_value)
    result_sum = None
    for _ in range(opt.ptb_n):
        err = np.random.randn(len(shape_params)) * opt.ptb_scale
        result = f(shape_params=shape_params + err)
        if result_sum is None:
            result_sum = result
        else:
            result_sum = result_sum + result
    return result_sum / opt.ptb_n


def fun(shape_params, lookup, opt):
    f = functools.partial(func, ret_value='func', lookup=lookup, opt=opt)
    return random_ptb(f=f, shape_params=shape_params, opt=opt)


def dis(shape_params, lookup, opt):
    f = functools.partial(func, ret_value='disp', lookup=lookup, opt=opt)
    return random_ptb(f=f, shape_params=shape_params, opt=opt)


def jac(shape_params, lookup, opt):
    f = functools.partial(func, ret_value='grad', lookup=lookup, opt=opt)
    return random_ptb(f=f, shape_params=shape_params, opt=opt)


def list2str(x):
    return ', '.join([str(item) for item in x])


def optimize():
    dolfin.set_log_active(False)
    opt = parse_args()
    x0 = np.array(opt.init_shape_params)
    left_shape = get_shape(side='left', shape_params=x0, opt=opt)
    # left_shape.visualize()
    right_shape = get_shape(side='right', shape_params=x0, opt=opt)
    # right_shape.visualize()

    lookup = dict()
    # print grad
    # get_mesh(side='left', opt=opt, vis=True)
    # get_mesh(side='right', opt=opt, vis=True)
    # plot_joint(left=left_shape, right=right_shape)
    # print(func(shape_params=x0, ret_value='disp', lookup=lookup, opt=opt, vis_grad=False, vis_mesh=True))
    # fun(shape_params=x0, opt=opt)
    # jac(shape_params=x0, opt=opt)
    # grad = func(control_points=x0, ret_value='grad', opt=opt, vis_grad=False, vis_mesh=False)
    # plot_grad(grad=grad, opt=opt)
    # print(grad)
    #
    # for eps in np.linspace(-.1, .1, 5):
    #     x0[0] = x0[0] + eps
    #     grad = func(control_points=x0, ret_value='grad', opt=opt, vis_grad=False, vis_mesh=False)
    #     print(grad)
    #     x0[0] = x0[0] - eps
    # quit()

    # grad test
    # print(func(control_points=x0, ret_value='grad', opt=opt))
    # delta = 1e-3
    # for idx in range(len(x0)):
    #     x0[idx] = x0[idx] + delta
    #     disp_p = func(control_points=x0, ret_value='disp', opt=opt)
    #     x0[idx] = x0[idx] - 2. * delta
    #     disp_l = func(control_points=x0, ret_value='disp', opt=opt)
    #     x0[idx] = x0[idx] + delta
    #     print((disp_p - disp_l) / delta / 2.)
    # quit()

    x = x0
    best = None
    best_x = None
    for _ in range(opt.gd_iter):
        grad = jac(shape_params=x, lookup=lookup, opt=opt)
        amax = None

        # ad hoc amax for `simple_joint`
        if opt.shape_name == 'simple_joint':
            amax = abs(((left_shape.h - 2.) / 2. - x[0]) / (-grad[0]))
        search_result = scipy.optimize.line_search(f=fun, myfprime=jac, xk=x, pk=-grad, gfk=grad, args=[lookup, opt],
                                                   c1=opt.ls_c1, c2=opt.ls_c2, amax=amax)
        alpha = search_result[0]
        if alpha is None:
            alpha = np.random.rand() * opt.gd_lr
        x = x - alpha * grad
        print(list2str(x))
        value = dis(shape_params=x, lookup=lookup, opt=opt)
        if best is None or value < best:
            best = value
            best_x = np.array(x)
            print(list2str(best_x), best)
    res_x = best_x
    print('x0:', list2str(x0))
    print('result:', list2str(res_x))
    opt.control_points = res_x
    get_shape(side='left', shape_params=res_x, opt=opt).visualize()
    get_shape(side='right', shape_params=res_x, opt=opt).visualize()
    print('disp', dis(shape_params=res_x, lookup=lookup, opt=opt))


def get_disp(opt):
    left = FEM(side='left', shape_params=opt.init_shape_params, opt=opt)
    right = FEM(side='right', shape_params=opt.init_shape_params, opt=opt)
    left_disp, right_disp = None, None
    for it in range(opt.num_it):
        if it < opt.num_it - 1:
            left.solve(right, opt=opt)
            right.solve(left, opt=opt)
        else:
            left_disp = left.solve(right, need_disp=True, opt=opt)
            right_disp = right.solve(left, need_disp=True, opt=opt)
    disp = right_disp - left_disp
    return disp, left, right


def adjoint_grad_check(query_list, scale=10.):
    dolfin.set_log_active(False)
    opt = parse_args()
    init_disp, left, right = get_disp(opt)
    left_grad = dolfin_adjoint.compute_gradient(init_disp, left.adjoint_ctrl)
    right_grad = dolfin_adjoint.compute_gradient(init_disp, right.adjoint_ctrl)
    dolfin_plot(left_grad, dpi=500, scale=1.5, xlim=(15., 35.), ylim=(-.5, 13.), need_colorbar=True)
    dolfin_plot(right_grad, dpi=500, scale=1.5, xlim=(10., 30.), ylim=(-.5, 13.), need_colorbar=True)
    adj_grad = np.zeros(2)
    delta = 1e-4
    for side, mesh, grad_field in [('left', left.mesh, left_grad), ('right', right.mesh, right_grad)]:
        print('side', side)
        finite_diff_list = []
        adjoint_list = []
        for query_vtx in tqdm(query_list):
            print('query_vtx', query_vtx)
            grad_field.eval(adj_grad, x=query_vtx)
            print('adj_grad', adj_grad)
            adjoint_list.append([adj_grad[0], adj_grad[1]])
            opt.enable_offset = side
            opt.target_vtx = query_vtx

            opt.offset = [delta / 2., 0.]
            disp_plus, _, _ = get_disp(opt)
            opt.offset = [-delta / 2., 0.]
            disp_minus, _, _ = get_disp(opt)
            finite_diff_x = (float(disp_plus) - float(disp_minus)) / delta
            print(finite_diff_x)

            opt.offset = [0., delta / 2.]
            disp_plus, _, _ = get_disp(opt)
            opt.offset = [0., -delta / 2.]
            disp_minus, _, _ = get_disp(opt)
            finite_diff_y = (float(disp_plus) - float(disp_minus)) / delta
            print(finite_diff_y)
            finite_diff_list.append([finite_diff_x, finite_diff_y])
        print(finite_diff_list, adjoint_list)
        for grad_name, grad_list in [('finite_diff', finite_diff_list), ('adjoint', adjoint_list)]:
            plt.figure(dpi=500)
            plt.gca().set_aspect('equal', 'box')
            plt.xlim(15., 28.)
            dolfin.plot(mesh)
            for idx in range(len(query_list)):
                query_point = query_list[idx]
                grad = grad_list[idx]
                plt.arrow(query_point[0], query_point[1], grad[0] * scale, grad[1] * scale, width=.03, color='black')
            plt.savefig('{}_{}.pdf'.format(side, grad_name))
            plt.show()
        adjoint_list = np.array(adjoint_list)
        finite_diff_list = np.array(finite_diff_list)
        print(np.abs((adjoint_list - finite_diff_list) / finite_diff_list))
        print(np.average(np.abs((adjoint_list - finite_diff_list) / finite_diff_list)))


def search():
    dolfin.set_log_active(False)
    opt = parse_args()
    best, best_x = None, None
    lookup = dict()
    for _ in tqdm(range(100)):
        while True:
            a = np.random.uniform(11.5, 11.7)
            b = np.random.uniform(3.4, 3.6)
            c = np.random.uniform(7.3, 7.5)
            if a > c:
                break
        x0 = np.array([a, b, c])
        try:
            disp = func(shape_params=x0, ret_value='disp', lookup=lookup, opt=opt, vis_grad=False, vis_mesh=False)
        except RuntimeError:
            continue
        if best is None or disp < best:
            best = disp
            best_x = x0
            print(x0, disp)
    print(best_x, best)
    opt.control_points = best_x
    get_mesh(side='left', opt=opt, vis=True)
    get_mesh(side='right', opt=opt, vis=True)


if __name__ == '__main__':
    # search()
    optimize()

    # query_list = [
    #     [20.000000000000000, 8.000000000000000],
    #     [22.000000000000000, 8.000000000000000],
    #     [23.000000000000000, 4.000000000000000],
    #     [26.000000000000000, 7.000000000000000],
    #     [20.500000000000000, 8.000000000000000],
    #     [21.000000000000000, 8.000000000000000],
    #     [21.500000000000000, 8.000000000000000],
    #     [22.111111111111111, 7.555555555555555],
    #     [22.222222222222221, 7.111111111111111],
    #     [22.333333333333332, 6.666666666666667],
    #     [22.444444444444443, 6.222222222222222],
    #     [22.555555555555557, 5.777777777777778],
    #     [22.666666666666668, 5.333333333333334],
    #     [22.777777777777779, 4.888888888888889],
    #     [22.888888888888889, 4.444444444444445],
    #     [23.333333333333332, 4.333333333333333],
    #     [23.666666666666668, 4.666666666666667],
    #     [24.000000000000000, 5.000000000000000],
    #     [24.333333333333332, 5.333333333333333],
    #     [24.666666666666668, 5.666666666666666],
    #     [25.000000000000000, 6.000000000000000],
    #     [25.333333333333332, 6.333333333333333],
    #     [25.666666666666668, 6.666666666666666],
    # ]
    # adjoint_grad_check(query_list=query_list)

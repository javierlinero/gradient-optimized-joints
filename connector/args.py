import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # shape
    parser.add_argument('--shape_name', type=str, default='simple_joint',
                        choices=['simple_joint', 'single_joint', 'double_joint'],
                        help='name of the shape')
    parser.add_argument('--init_shape_params', type=float, default=[10.795044579355393, 3.4424377831583315, 8.302111706576671],
                        help='initial shape parameters')
    parser.add_argument('--mesh_size', type=float, default=.5,
                        help='step size of the mesh')
    parser.add_argument('--enable_offset', type=str, default=None, choices=[None, 'left', 'right'],
                        help='side to apply offset [None, left, right]')
    parser.add_argument('--target_vtx', type=float, default=None,
                        help='target vertex to move')
    parser.add_argument('--offset', type=float, default=None,
                        help='offset of the vertex to move')

    # fem-related
    parser.add_argument('--young_modulus', type=float, default=1.,
                        help='Young''s modulus')
    parser.add_argument('--poisson_ratio', type=float, default=.4,
                        help='Poisson''s ratio')
    parser.add_argument('--traction', type=float, default=.001,
                        help='magnitude of the tractive force')
    parser.add_argument('--pen_w', type=float, default=1.,
                        help='weight of the penalization term')
    parser.add_argument('--soft_relu_scale', type=float, default=50.,
                        help='scale of the soft relu')
    parser.add_argument('--num_it', type=int, default=4,
                        help='number of iterations in the alternating solver')
    parser.add_argument('--penalization_type', type=str, default='line_fitting', choices=['line_fitting', 'sdf'],
                        help='what penalization type to use for collision detection')

    # optimization
    parser.add_argument('--ctrl_var', type=str, default='mesh_offset',
                        choices=['spatial_coordinate', 'mesh_offset'],
                        help='control variable selection for dolfin-adjoint')
    parser.add_argument('--gd_iter', type=int, default=15,
                        help='number of gradient descent iterations')
    parser.add_argument('--gd_lr', type=float, default=.5,
                        help='learning rate when line search fails')
    parser.add_argument('--ls_c1', type=float, default=1e-4,
                        help='c1 for strong Wolfe conditions in the line search')
    parser.add_argument('--ls_c2', type=float, default=.9,
                        help='c2 for strong Wolfe conditions in the line search')
    parser.add_argument('--incl_endpts', type=bool, default=True,
                        help='include endpoints during gradient calculation')
    parser.add_argument('--ptb_n', type=int, default=3,
                        help='number of the random perturbations applied')
    parser.add_argument('--ptb_scale', type=float, default=.01,
                        help='scale of the random perturbations applied')
    parser.add_argument('--reg_w', type=float, default=1.,
                        help='weight for the regularizer')
    parser.add_argument('--reg_soft_min_len', type=float, default=1.5,
                        help='soft minimum length for the contacting length')
    parser.add_argument('--reg_soft_min_width', type=float, default=2.5,
                        help='soft minimum width for the left-hand side')

    # misc
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='epsilon')
    parser.add_argument('--disp_save_to', type=str, default='output/displacement.pvd',
                        help='displacement saving path')

    opt = parser.parse_args()
    return opt

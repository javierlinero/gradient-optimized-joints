import json
from shapes import get_shape 
from visualization import dolfin_plot
from args import parse_args

if __name__ == '__main__':
    opt = parse_args()

    with open('shape_dat.json') as f:
        data = json.load(f)
    
    for shape in data['shapes']:
        opt.shape_name = shape['shape_name']
        opt.init_shape_params = shape['init_shape_params']

        l = get_shape(side='left', shape_params=opt.init_shape_params, opt=opt)
        r = get_shape(side='right', shape_params=opt.init_shape_params, opt=opt)
        dolfin_plot(l.mesh)
        dolfin_plot(r.mesh)

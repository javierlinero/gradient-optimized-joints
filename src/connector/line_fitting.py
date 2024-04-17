import torch
import dolfin
import pyadjoint.overloaded_function
import dolfin_adjoint


def backend_line_fitting(func, idx_list, adj_inputs=None):
    vtx_list = torch.tensor(func.vector()[:], requires_grad=True, dtype=torch.float64)
    vtx_2d = torch.reshape(vtx_list, (-1, 2))
    data = vtx_2d[idx_list]
    x_bar = torch.mean(data[:, 0])
    y_bar = torch.mean(data[:, 1])
    beta = torch.sum((data[:, 0] - x_bar) * (data[:, 1] - y_bar)) / \
        torch.sum((data[:, 0] - x_bar) * (data[:, 0] - x_bar))
    alpha = y_bar - beta * x_bar
    if adj_inputs is None:
        return dolfin_adjoint.Constant(alpha.item()), dolfin_adjoint.Constant(beta.item())
    else:
        assert len(adj_inputs) == 2
        vtx_list.retain_grad()
        alpha.backward(gradient=torch.tensor(adj_inputs[0]), retain_graph=True)
        beta.backward(gradient=torch.tensor(adj_inputs[1]))
        grad = vtx_list.grad.cpu().detach().numpy()
        grad_vec = dolfin.cpp.la.PETScVector(dolfin.MPI.comm_world, func.vector().local_size())
        grad_vec.vec()[:] = grad
        return grad_vec


class LineFittingBlock(pyadjoint.Block):
    def __init__(self, func, idx_list, **kwargs):
        super(LineFittingBlock, self).__init__()
        self.idx_list = idx_list
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'LineFittingBlock'

    def recompute_component(self, inputs, block_variable, idx, prepared):
        assert len(inputs) == 1 and idx in [0, 1]
        return backend_line_fitting(inputs[0], self.idx_list)[idx]

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        assert len(inputs) == 1 and len(adj_inputs) == 2 and idx == 0
        return backend_line_fitting(inputs[0], self.idx_list, (adj_inputs[0].get_local()[0],
                                                               adj_inputs[1].get_local()[0]))

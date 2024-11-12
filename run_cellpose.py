from cellpose import utils, models, io


def cellpose_func(cell_im, cellpose_model, path):
    model = models.CellposeModel(gpu=False, pretrained_model=path+cellpose_model)
    chan = [0,0]
    masks, flows, styles = model.eval(cell_im, diameter=None, invert=False, channels=chan)
    return masks


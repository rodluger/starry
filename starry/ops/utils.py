# -*- coding: utf-8 -*-

__all__ = ["infer_size"]


def infer_size(tensor):
    # TODO: Need a smarter way of inferring the shape of tensors!
    try:
        return tensor.tag.test_value.shape[0]
    except:
        try:
            return tensor.shape.eval()[0]
        except:
            raise Exception("Unable to infer the size of tensor %s." % tensor)
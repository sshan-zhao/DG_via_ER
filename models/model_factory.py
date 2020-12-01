from models import resnet

nets_map = {
'resnet18': resnet.resnet18,
'resnet50': resnet.resnet50,
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn

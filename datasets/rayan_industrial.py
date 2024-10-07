from datasets.mvtec import MVTecDataset

_CLASSNAMES = [
    'capsule',
    'capsules',
    'capsules_squared',
    'juice_bottle',
    'juice_bottle_squared',
    'macaroni2',
    'macaroni2_squared',
    'pcb3',
    'pcb3_squared',
    'photovoltaic_module',
    'pill',
    'pushpins',
    'pushpins_squared',
    'screw_bag',
    'screw_bag_squared',
    'turbine',
]


class RayanIndustrialDataset(MVTecDataset):
    def __init__(self, source, classname, **kwargs):
        super().__init__(source, classname, **kwargs)
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

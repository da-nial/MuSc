from datasets.mvtec import MVTecDataset

_CLASSNAMES = [
    'BrainMRI',
    'LiverCT',
    'Retina_RESC',
    'Retina_RESC_squared',
]


class RayanMedicalDataset(MVTecDataset):
    def __init__(self, source, classname, **kwargs):
        super().__init__(source, classname, **kwargs)
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

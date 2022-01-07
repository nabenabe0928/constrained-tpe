from typing import List, Optional

from targets.hpolib import api as hpolib
from targets.nasbench101 import api as nasbench101
from targets.nasbench201 import api as nasbench201
from util.constants import DOMAIN_SIZE_CHOICES


def find_oracle(constraints: List[str], feasible_domain: Optional[int] = None) -> None:
    print(f"\n### {feasible_domain}% ###")
    cs = constraints
    cstrs = [getattr(hpolib.ConstraintChoices, c) for c in cs]

    for dataset in hpolib.DatasetChoices:
        bm = hpolib.HPOBench(
            dataset=dataset,
            feasible_domain_ratio=feasible_domain,
            constraints=cstrs
        )
        if bm.oracle is None:
            print(dataset, bm.find_oracle())

    cstrs = [getattr(nasbench101.ConstraintChoices, c) for c in cs]
    bm = nasbench101.NASBench101(feasible_domain_ratio=feasible_domain, constraints=cstrs)
    if bm.oracle is None:
        print('NASBench101', bm.find_oracle())

    cs = ['size_in_mb' if c == 'n_params' else c for c in cs]
    cstrs = [getattr(nasbench201.ConstraintChoices, c) for c in cs]

    for dataset in nasbench201.DatasetChoices:
        bm = nasbench201.NASBench201(
            dataset=dataset,
            feasible_domain_ratio=feasible_domain,
            constraints=cstrs
        )
        if bm.oracle is None:
            print(dataset, bm.find_oracle())


if __name__ == '__main__':
    constraint_choices = [['n_params'], ['runtime'], ['runtime', 'n_params']]
    for cs in constraint_choices:
        print(cs)
        for feasible_domain in DOMAIN_SIZE_CHOICES:
            find_oracle(constraints=cs, feasible_domain=feasible_domain)

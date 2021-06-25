import os
import mobie
import numpy as np
import elf.transformation as etrafo
from pybdv.metadata import get_affine, write_affine


def move_trafo(source_name):
    ds_folder = './data/yeast'
    ds_metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    assert source_name in ds_metadata['sources']

    views = ds_metadata['views']
    view = views[source_name]

    if 'sourceTransforms' in view:
        return

    def update_xml(xml):
        assert os.path.exists(xml), xml

        trafo = get_affine(xml, setup_id=0)['affine0']
        trafo = etrafo.parameters_to_matrix(trafo)

        scale = etrafo.affine.affine_matrix_3d(scale=etrafo.affine.scale_from_matrix(trafo))
        inv_scale = np.linalg.inv(scale)
        trafo2 = inv_scale @ trafo
        assert np.allclose(scale @ trafo2, trafo)
        trafo2 = etrafo.matrix_to_parameters(trafo2)

        scale = etrafo.matrix_to_parameters(scale)
        write_affine(xml, setup_id=0, affine=scale, overwrite=True)

        return trafo2

    xml_path1 = f"./data/yeast/images/local/{source_name}.xml"
    trafo = update_xml(xml_path1)

    xml_path2 = f"./data/yeast/images/remote/{source_name}.xml"
    trafo2 = update_xml(xml_path2)
    assert np.allclose(trafo, trafo2), f"{trafo}, {trafo2}"

    # update the view
    view['sourceTransforms'] = [
        {"affine": {"parameters": trafo, "sources": [source_name]}}
    ]
    views[source_name] = view
    ds_metadata['views'] = views

    mobie.metadata.write_dataset_metadata(ds_folder, ds_metadata)


def move_all_trafos():
    ds_folder = './data/yeast'
    sources = mobie.metadata.read_dataset_metadata(ds_folder)['sources']
    for source_name in sources:
        if 'tomogram' not in source_name:
            continue
        move_trafo(source_name)


if __name__ == '__main__':
    move_all_trafos()

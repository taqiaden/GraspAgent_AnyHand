import coacd
import trimesh
import os
import xml.etree.ElementTree as ET


def mesh_to_parts(obj_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        return

    mesh = trimesh.load(obj_path, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    parts = coacd.run_coacd(
        mesh,
        threshold=0.05,  # 精度阈值，默认: 0.05
        max_convex_hull=-1,  # 最大凸包数量，默认: -1 (无限制)
        preprocess_mode="auto",  # 预处理模式，默认: "auto"
        preprocess_resolution=30,  # 预处理分辨率，默认: 30
        resolution=2000,  # 分解分辨率，默认: 2000
        mcts_nodes=20,  # MCTS节点数，默认: 20
        mcts_iterations=150,  # MCTS迭代次数，默认: 150
        mcts_max_depth=3,  # MCTS最大深度，默认: 3
        pca=False,  # 是否启用PCA降维，默认: False
        merge=True,  # 是否合并小凸包，默认: True
        decimate=False,  # 是否简化凸包，默认: False
        max_ch_vertex=256,  # 每个凸包的最大顶点数，默认: 256
        extrude=False,  # 是否拉伸凸包，默认: False
        extrude_margin=0.01,  # 拉伸边距，默认: 0.01
        apx_mode="ch",  # 近似模式，默认: "ch" (凸包)
        seed=0  # 随机种子，默认: 0
    )

    for i, part in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
        part_mesh.export(os.path.join(output_dir, f"part_{i}.obj"))

    print(f"Exported {len(parts)} convex parts to {output_dir}")


def gen_xml(mesh_name, output_dir):
    tree = ET.parse('./mesh_template.xml')
    root = tree.getroot()
    root.attrib['model'] = mesh_name

    asset = root.find('asset')
    mesh = ET.Element('mesh')
    mesh.set('name', mesh_name)
    mesh.set('file', "object/" + mesh_name + "/model.obj")
    asset.append(mesh)

    body = root.find('worldbody').find('body')
    geom = ET.Element('geom')
    geom.set('name', mesh_name)
    geom.set('class', "object/visual")
    geom.set('mesh', mesh_name)
    body.append(geom)

    num_parts = len(os.listdir(output_dir))
    print(num_parts)
    for i in range(num_parts):
        mesh = ET.Element('mesh')
        mesh.set('name', mesh_name + '_' + str(i))
        mesh.set('file', "object/" + mesh_name + "/output_parts/part_" + str(i) + ".obj")
        asset.append(mesh)

        geom = ET.Element('geom')
        geom.set('name', mesh_name + '_' + str(i))
        geom.set('class', "object/collision")
        geom.set('mesh', mesh_name + '_' + str(i))
        body.append(geom)

    tree.write('../shadow_dexee/mesh/' + mesh_name + '.xml')


if __name__ == '__main__':
    objects_path = "../shadow_dexee/assets/object"
    files = os.listdir(objects_path)
    n = len(files)
    for i in range(n):
        mesh_name = "mesh_"+str(i)
        obj_path = os.path.join(objects_path, mesh_name, "model.obj")
        print(obj_path)
        output_dir = os.path.join(objects_path, mesh_name, "output_parts/")
        print(output_dir)
        mesh_to_parts(obj_path, output_dir)

        gen_xml(mesh_name, output_dir)
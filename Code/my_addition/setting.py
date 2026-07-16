import os
import argparse

cur_path = os.path.abspath(__file__)
project_root = cur_path
for _ in range(3):
    project_root = os.path.dirname(project_root)

inter_path = os.path.join("DRIVE", "test", "unet", "vseg")
output_path = os.path.join(project_root, "Output", inter_path)

parser = argparse.ArgumentParser(description="Parameters of the project")

"""output path"""
parser.add_argument("--dataset_name", type=str, default="CFD")
parser.add_argument("--output_path", type=str, default=output_path)
parser.add_argument("--refine_seg_path", type=str, default=os.path.join(output_path, "refine_seg"))
parser.add_argument("--marked_refine_path", type=str, default=os.path.join(output_path, "marked_refine"))
parser.add_argument("--refine_seg_color_path", type=str, default=os.path.join(output_path, "refine_seg_color"))
parser.add_argument("--bar_img_path", type=str, default=os.path.join(output_path, "bar_img"))
parser.add_argument("--video_path", type=str, default=os.path.join(output_path, "videos"))


"""other settings"""
parser.add_argument("--search_radius", type=int, default=10) # CFD:10, crack-seg:30, DeepCrack:30

"""ablation study"""
parser.add_argument("--ext_2_valid", type=bool, default=True, help="whether to use ext_2_valid")
parser.add_argument("--repel_valid", type=bool, default=True, help="whether to use repel_valid")
parser.add_argument("--gra_valid", type=bool, default=True, help="whether to use gra_valid")

parser.add_argument("--win", type=int, default=5)

add_args = parser.parse_args()

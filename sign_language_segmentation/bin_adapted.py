#!/usr/bin/env python
import argparse
import os

import numpy as np
import pympi
import sys
import torch
from src.model import PoseTaggingModel
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs, normalize_hands_3d

from sign_language_segmentation.src.utils.probs_to_segments import probs_to_segments


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'  # Convert s to lowercase before comparison


def add_optical_flow(pose: Pose):
    from pose_format.numpy.representation.distance import DistanceRepresentation
    from pose_format.utils.optical_flow import OpticalFlowCalculator

    print("\nAdding optical flow to pose (in bin.add_optical_flow)…\n")

    print(f"Original pose shape (in bin.add_optical_flow):\t{pose.body.data.shape}\n")

    calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())
    flow = calculator(pose.body.data)  # numpy: frames - 1, people, points
    flow = np.expand_dims(flow, axis=-1)  # frames - 1, people, points, 1
    # add one fake frame in numpy
    flow = np.concatenate([np.zeros((1, *flow.shape[1:]), dtype=flow.dtype), flow], axis=0)

    print(f"Optical flow shape (in bin.add_optical_flow):\t{flow.shape}\n")

    # Add flow data to X, Y, Z
    pose.body.data = np.concatenate([pose.body.data, flow], axis=-1).astype(np.float32)

    print(f"Final pose shape (in bin.add_optical_flow):\t{pose.body.data.shape}\n")


def process_pose(pose: Pose, optical_flow=False, hand_normalization=False):
    print(f"Optical flow variable value (in bin.process_pose):\t{optical_flow}")

    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info)
    pose_hide_legs(pose)

    if hand_normalization:
        normalize_hands_3d(pose)

    if optical_flow:
        print("Adding optical flow to pose (in bin.process_pose)")
        add_optical_flow(pose)

    print(f"\npose.body.data.shape (in bin.process_pose):\t{pose.body.data.shape}\n")

    return pose


def load_model(model_path: str):
    model = torch.jit.load(model_path)
    model.eval()
    print("Model loaded from:", model_path)
    print(f"model:\t{str(model)}")
    print("Model code:", model.code)
    return model


def load_model_no_jit(model_path: str, model_args):
    # Initialize the model with the required architecture details
    model = PoseTaggingModel(**model_args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded from:", model_path)
    print(f"model:\t{str(model)}")
    return model


def load_model_no_jit_2(model_path, model_args):
    # Load the entire checkpoint, not just the state_dict
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    print(f"Model args (in bin.load_model_no_jit_2):\t{model_args}")

    # Instantiate the model using the provided arguments
    print("About to load model as PoseTaggingModel in bin.load_model_no_jit_2")
    model = PoseTaggingModel(**model_args)

    # Extract the state_dict specifically for the model
    print("About to extract state_dict from checkpoint in bin.load_model_no_jit_2")
    model_state_dict = checkpoint['state_dict']

    # Adjust the keys in the state_dict by removing the 'model.' prefix added by PyTorch Lightning
    print("About to adjust the model state_dict in bin.load_model_no_jit_2")
    adjusted_model_state_dict = {key.replace("model.", ""): value for key, value in model_state_dict.items()}

    # print(f"\nadjusted_model_state_dict:\n")

    # print("\n\n\n")
    # for key, value in adjusted_model_state_dict.items():
        # print(f"key: {key}\tvalue: {value}")
    # print("\n\n\n")

    # Load the adjusted state_dict into the model
    print("About to load the adjusted model state_dict in bin.load_model_no_jit_2")
    model.load_state_dict(adjusted_model_state_dict)

    print("About to return the model in bin.load_model_no_jit_2")
    return model


def predict(model, pose: Pose):
    with torch.no_grad():
        torch_body = pose.body.torch()
        pose_data = torch_body.data.tensor[:, 0, :, :].unsqueeze(0)

        print(f"Input pose_data shape to model: {pose_data.shape}")

        return model(pose_data)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--elan', required=True, type=str, help='path to output elan file')
    parser.add_argument('--filename', default=None, required=False, type=str, help='path to filename file')
    parser.add_argument('--subtitles', default=None, required=False, type=str, help='path to subtitle file')
    parser.add_argument('--model', default='model_E1s-1.pth', required=False, type=str, help='path to model file')
    parser.add_argument('--no-pose-link', action='store_true', help='whether to link the pose file')
    parser.add_argument('--jit', type=boolean_string, default=False, help='whether to save model without code?')
    parser.add_argument('--non_jit', type=boolean_string, default=True, help='whether to save model with code?')
    parser.add_argument('--optical_flow', type=boolean_string, default=False,
                        help='whether the model uses optical flow')
    parser.add_argument('--use_i_threshold', type=boolean_string, default=False, help='whether to use i_threshold')

    parser.add_argument('--sign_segments_b_threshold', type=int, default=35, required=False,
                        help='Sign segments b_threshold')
    parser.add_argument('--sign_segments_i_threshold', type=int, default=0.5, required=False,
                        help='Sign segments i_threshold')
    parser.add_argument('--sign_segments_o_threshold', type=int, default=30, required=False,
                        help='Sign segments o_threshold')
    parser.add_argument('--sentence_segments_b_threshold', type=int, default=40, required=False,
                        help='Sentence segments b_threshold')
    parser.add_argument('--sentence_segments_i_threshold', type=int, default=0.5, required=False,
                        help='Sentence segments i_threshold')
    parser.add_argument('--sentence_segments_o_threshold', type=int, default=90, required=False,
                        help='Sentence segments o_threshold')

    return parser.parse_args()


def main():
    args = get_args()

    print('Loading pose ...')
    with open(args.pose, "rb") as f:
        pose = Pose.read(f.read())
        if 'E4' in args.model:
            pose = process_pose(pose, optical_flow=True, hand_normalization=True)
        else:
            if args.optical_flow:
                pose = process_pose(pose, optical_flow=True)
            else:
                pose = process_pose(pose)

    # print(f'Optical Flow: {"Yes" if "E4" in args.model else "No"}')

    print(f'Hand Normalization: {"Yes" if "E4" in args.model else "No"}')

    print('Loading model ...')
    install_dir = str(os.path.dirname(os.path.abspath(__file__)))
    print(f"install_dir:\t{install_dir}")

    if args.jit:
        model = load_model(os.path.join(install_dir, "..", "..", "models", args.model))
    else:
        if args.non_jit:

            model_args = {'sign_class_weights': [7.105956030624457, 276.6916930933652, 1.1686900665577535],
                          'sentence_class_weights': [15.605907542499951, 11869.878091872792, 1.0685616341999475],
                          'pose_dims': (75, 3), 'pose_projection_dim': 256, 'hidden_dim': 512, 'encoder_depth': 1,
                          'encoder_bidirectional': True, 'encoder_autoregressive': False, 'tagset_size': 3,
                          'lr_scheduler': 'none', 'learning_rate': 0.001, 'b_threshold': 50, 'o_threshold': 50,
                          'threshold_likeliest': False}  # new_loss_fn_02_validation_sign_segment_IoU
            model = load_model_no_jit_2(os.path.join(install_dir, "..", "..", "models", args.model, "best.ckpt"), model_args)

        else:
            print("Please specify whether to load jit or non-jit model")
            # sys.exit(1)

    print(f'\nModel:\t{str(model)}\n')
    print(f'\nPose:\t{str(pose)}\n')

    print('Estimating segments ...')

    print(f'Processed Pose Shape: {pose.body.data.shape}')

    probs = predict(model, pose)

    print(f"args.sign_segments_b_threshold: {args.sign_segments_b_threshold}")
    print(f"args.sign_segments_o_threshold: {args.sign_segments_o_threshold}")
    print(f"args.sentence_segments_b_threshold: {args.sentence_segments_b_threshold}")
    print(f"args.sentence_segments_o_threshold: {args.sentence_segments_o_threshold}")
    print(f"args.sign_segments_i_threshold: {args.sign_segments_i_threshold}")
    print(f"args.sentence_segments_i_threshold: {args.sentence_segments_i_threshold}")

    if not args.use_i_threshold:

        sign_segments = probs_to_segments(probs["sign"], args.sign_segments_b_threshold, args.sign_segments_o_threshold)
        sentence_segments = probs_to_segments(probs["sentence"], args.sentence_segments_b_threshold,
                                              args.sentence_segments_o_threshold)

    else:
        print("I haven't enabled decoding on just the 'i' threshold yet.  Exiting.  Don't use that.")
        sys.exit(1)

    print(f"sign_segments:\n{str(sign_segments)}\n\n")

    print('Building ELAN file ...')
    tiers = {
        "SIGN": sign_segments,
        "SENTENCE": sentence_segments,
    }

    fps = pose.body.fps

    print(f"\nfps:\t{str(fps)}\n\n")

    eaf = pympi.Elan.Eaf(author="sign-langauge-processing/transcription")
    # if args.video is not None:
    #     mimetype = None  # pympi is not familiar with mp4 files
    #     if args.video.endswith(".mp4"):
    #         mimetype = "video/mp4"
    #     eaf.add_linked_file(args.video, mimetype=mimetype)

    if args.filename is not None:
        mimetype = None  # pympi is not familiar with mp4 files
        if args.filename.endswith(".mp4"):
            mimetype = "video/mp4"
        eaf.add_linked_file(args.filename, mimetype=mimetype)

    if not args.no_pose_link:
        eaf.add_linked_file(args.pose, mimetype="application/pose")

    for tier_id, segments in tiers.items():
        eaf.add_tier(tier_id)
        for segment in segments:
            eaf.add_annotation(tier_id, int(segment["start"] / fps * 1000), int(segment["end"] / fps * 1000))

    if args.subtitles and os.path.exists(args.subtitles):
        import srt
        eaf.add_tier("SUBTITLE")
        with open(args.subtitles, "r") as infile:
            for subtitle in srt.parse(infile):
                start = subtitle.start.total_seconds()
                end = subtitle.end.total_seconds()
                eaf.add_annotation("SUBTITLE", int(start * 1000), int(end * 1000), subtitle.content)

    print('Saving to disk…')
    eaf.to_file(args.elan)


if __name__ == '__main__':
    main()

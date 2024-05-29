#!/usr/bin/env python
import argparse
import numpy as np
import os
import pympi
import sys
import torch
from copy import deepcopy
from src.model import PoseTaggingModel
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs, normalize_hands_3d
from pickle_utils import save_to_pickle
from scipy import stats
from sign_language_segmentation.src.utils.probs_to_segments import (probs_to_segments, custom_probs_to_segments,
                                                                    custom_probs_to_segments_with_o,
                                                                    custom_probs_to_segments_simple,
                                                                    custom_probs_to_segments_simple_2)


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
    parser.add_argument('--id', required=True, type=str, help='the base name of the input file')
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--elan_milliseconds_dir', required=True, type=str,
                        help='path to output elan directory, with milliseconds')
    parser.add_argument('--elan_frames_dir', required=True, type=str, help='path to output elan directory, with frames')
    parser.add_argument('--segments_pickle_dir', required=False, type=str,
                        help='path to output segmentation directory, with frames; pickle format')
    parser.add_argument('--segments_plaintext_dir', required=False, type=str,
                        help='path to output segmentation directory, with frames; plaintext format')
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


def create_eaf(args, tiers, fps, boundary_type, postprocessed):
    assert boundary_type in ["milliseconds", "frames"]

    eaf = pympi.Elan.Eaf(author="sign-language-processing/transcription")

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
            if boundary_type == "milliseconds":
                # print(f"segment['start']:\t{str(segment['start'])}")
                # print(f"segment['end']:\t{str(segment['end'])}\n")
                eaf.add_annotation(tier_id, int(segment["start"] / fps * 1000), int(segment["end"] / fps * 1000))
            elif boundary_type == "frames":
                # print(f"segment['start']:\t{str(segment['start'])}")
                # print(f"segment['end']:\t{str(segment['end'])}")
                eaf.add_annotation(tier_id, int(segment["start"]), int(segment["end"]))

    print(
        f"Saving elan file to disk…\n\n================================================================================\n\n")

    if boundary_type == "frames":
        if postprocessed:
            eaf.to_file(os.path.join(args.elan_frames_dir, "POSTPROCESSED", str(args.id) + ".eaf"))
        else:
            eaf.to_file(os.path.join(args.elan_frames_dir, "NON_POSTPROCESSED", str(args.id) + ".eaf"))

    elif boundary_type == "milliseconds":
        if postprocessed:
            eaf.to_file(os.path.join(args.elan_milliseconds_dir, "POSTPROCESSED", str(args.id) + ".eaf"))
        else:
            eaf.to_file(os.path.join(args.elan_milliseconds_dir, "NON_POSTPROCESSED", str(args.id) + ".eaf"))


def get_mmm(data, tag):
    # "MMM" is short for "mean, median, mode" + related values

    BIO = {"O": 0, "B": 1, "I": 2}

    values = data[:, BIO[tag]]

    mean_val = np.mean(values)
    median_val = np.median(values)
    mode_val, _ = stats.mode(values)
    max_val = np.max(values)
    min_val = np.min(values)
    std_val = np.std(values)

    # Ensure mode_val is correctly handled
    if isinstance(mode_val, np.ndarray):
        mode_val = mode_val[0]

    mmm_data_structure = {"mean": mean_val, "median": median_val, "mode": mode_val, "max": max_val, "min": min_val,
                          "std": std_val}

    return mmm_data_structure


def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)


#######################################################################################################################

def calculate_threshold(tag, mean, median, min, max, std):
    print(f"tag:\t{tag}")
    print(f"mean:\t{mean}")
    print(f"min:\t{min}")
    print(f"max:\t{max}")
    print(f"std:\t{std}\n")

    if tag not in ["B", "I", "O"]:
        raise ValueError(f"Invalid tag: {tag}.  Expected one of 'B', 'I', or 'O'.")

    # This logic is only a heuristic, for now.
    if tag == "B" or tag == "I":

        # threshold = int(np.floor(mean - (0.5 * std)))
        # threshold = int(np.floor(mean + (0.5 * std)))
        threshold = int(np.floor(mean))

        if threshold < min:
            threshold = int(min)

    elif tag == "O":

        threshold = int(np.floor(mean) - (0.5 * std))
        # threshold = int(np.floor(mean + (0.5 * std)))
        # threshold = int(np.floor(mean))

        # threshold = max - (0.75 * std)

        # threshold = np.floor(mean + 5)

        # threshold = median
        # threshold = median + (0.2 * std)

        if threshold < min:
            threshold = int(min)

    return threshold


#######################################################################################################################

def postprocess_segments(start_and_end_frames, fps, endpoint, seconds_of_gap_to_combine, seconds_to_delete,
                         seconds_to_expand):
    print(f"endpoint:\t{str(endpoint)}")

    if not start_and_end_frames:
        return []

    def combine_segments(start_and_end_frames, gap_frames):
        comb_segs = []
        current_segment = start_and_end_frames[0]

        for segment in start_and_end_frames[1:]:
            if segment['start'] <= current_segment['end'] + gap_frames:
                current_segment['end'] = max(current_segment['end'], segment['end'])
            else:
                comb_segs.append(current_segment)
                current_segment = segment

        comb_segs.append(current_segment)
        return comb_segs

    # Initial combination of segments
    combined_segments = combine_segments(start_and_end_frames, seconds_of_gap_to_combine * fps)

    # Remove short segments
    combined_segments = [
        segment for segment in combined_segments
        if segment['end'] - segment['start'] >= (seconds_to_delete * fps)
    ]

    if not combined_segments:
        return []

    # Expand each segment
    expanded_segments = []
    for segment in combined_segments:
        expanded_segment = {
            'start': max(0, segment['start'] - int(seconds_to_expand * fps)),
            'end': min(segment['end'] + int(seconds_to_expand * fps), endpoint)
        }
        expanded_segments.append(expanded_segment)

    # Recombine expanded segments
    final_segments = combine_segments(expanded_segments, seconds_of_gap_to_combine * fps)

    # Sanity check
    for i in range(len(final_segments) - 1):
        if final_segments[i]['end'] >= final_segments[i + 1]['start']:
            raise ValueError(f"Invalid segments: Segment {i} ends at {final_segments[i]['end']}, which is not before "
                             f"segment {i + 1}, which starts at {final_segments[i + 1]['start']}.")

    return final_segments


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

            # The model_args values should be considered hardcoded; they come from the segmentation model training
            # new_loss_fn_02_validation_sign_segment_IoU
            # model_args = {'sign_class_weights': [7.105956030624457, 276.6916930933652, 1.1686900665577535],
            #               'sentence_class_weights': [15.605907542499951, 11869.878091872792, 1.0685616341999475],
            #               'pose_dims': (75, 3), 'pose_projection_dim': 256, 'hidden_dim': 512, 'encoder_depth': 1,
            #               'encoder_bidirectional': True, 'encoder_autoregressive': False, 'tagset_size': 3,
            #               'lr_scheduler': 'none', 'learning_rate': 0.001, 'b_threshold': 50, 'o_threshold': 50,
            #               'threshold_likeliest': False}

            # new_splits_08_validation_sign_segment_IoU
            # model_args = {'pose_dims': (75, 3), 'pose_projection_dim': 256, 'hidden_dim': 512, 'encoder_depth': 1,
            #               'encoder_bidirectional': True, 'encoder_autoregressive': False, 'learning_rate': 0.001,
            #               'lr_scheduler': 'none', 'b_threshold': 50, 'o_threshold': 50, 'threshold_likeliest': False,
            #               'sign_class_weights': [6.686953934377452, 270.10106528270967, 1.1809822728450752],
            #               'sentence_class_weights': [16.65377338894846, 11409.692307692309, 1.063981570449882]}

            # kylie_noflow_01
            # model_args = {'sign_class_weights': [2.2511743215031315, 313.6909090909091, 1.8096286972938955],
            #               'sentence_class_weights': [6.955919903238812, 1478.8285714285714, 1.1688232504572862],
            #               'pose_dims': (75, 4), 'pose_projection_dim': 256, 'hidden_dim': 512, 'encoder_depth': 1,
            #               'encoder_bidirectional': True, 'encoder_autoregressive': False, 'tagset_size': 3,
            #               'lr_scheduler': 'none', 'learning_rate': 0.001, 'b_threshold': 50, 'o_threshold': 50,
            #               'threshold_likeliest': False}

            # kylie_noflow_02
            # model_args = {'sign_class_weights': [2.2511743215031315, 313.6909090909091, 1.8096286972938955],
            #               'sentence_class_weights': [6.955919903238812, 1478.8285714285714, 1.1688232504572862],
            #               'pose_dims': (75, 4), 'pose_projection_dim': 256, 'hidden_dim': 512, 'encoder_depth': 1,
            #               'encoder_bidirectional': True, 'encoder_autoregressive': True, 'tagset_size': 3,
            #               'lr_scheduler': 'none', 'learning_rate': 0.001, 'b_threshold': 50, 'o_threshold': 50,
            #               'threshold_likeliest': False}

            # kylie_noflow_06
            # model_args = {'sign_class_weights': [2.2511743215031315, 313.6909090909091, 1.8096286972938955],
            #               'sentence_class_weights': [6.955919903238812, 1478.8285714285714, 1.1688232504572862],
            #               'pose_dims': (75, 4), 'pose_projection_dim': 256, 'hidden_dim': 128, 'encoder_depth': 2,
            #               'encoder_bidirectional': True, 'encoder_autoregressive': False, 'tagset_size': 3,
            #               'lr_scheduler': 'none', 'learning_rate': 0.001, 'b_threshold': 50, 'o_threshold': 50,
            #               'threshold_likeliest': False}

            # kylie_NOflow_01
            model_args = {'sign_class_weights': [2.2511743215031315, 313.6909090909091, 1.8096286972938955],
                          'sentence_class_weights': [6.955919903238812, 1478.8285714285714, 1.1688232504572862],
                          'pose_dims': (75, 3), 'pose_projection_dim': 256, 'hidden_dim': 512, 'encoder_depth': 1,
                          'encoder_bidirectional': True, 'encoder_autoregressive': False, 'tagset_size': 3,
                          'lr_scheduler': 'none', 'learning_rate': 0.001, 'b_threshold': 50, 'o_threshold': 50,
                          'threshold_likeliest': False}

            model = load_model_no_jit_2(os.path.join(install_dir, "..", "..", "models", args.model, "best.ckpt"),
                                        model_args)

        else:
            print("Please specify whether to load jit or non-jit model")

    print(f'\nModel:\t{str(model)}\n')
    print(f'\nPose:\t{str(pose)}\n')

    print('Estimating segments ...')

    print(f'Processed Pose Shape: {pose.body.data.shape}')

    probs = predict(model, pose)

    print(f"probs:\t{str(probs['sentence'])}")

    overall_len = probs["sentence"].shape[1] - 1
    print(f"overall_len:\t{str(overall_len)}")
    # sys.exit(1)

    fps = pose.body.fps
    print(f"\nfps:\t{str(fps)}\n")

    torch.set_printoptions(threshold=torch.inf)

    # print(f"logits (sentence) shape:\t{str(probs['sentence'].shape)}")
    # print(f"\nlogits (sentence):\n{str(probs['sentence'])}\n")

    # This model is not trained to accurately predict "sign segments", only "sentence segments".  Setting sign
    # segments to the empty list for simplicity.
    # sign_segments = probs_to_segments(probs["sign"], args.sign_segments_b_threshold, args.sign_segments_o_threshold)
    sign_segments = []

    sentence_probs_percentages = np.round(np.exp(probs["sentence"].numpy().squeeze()) * 100)
    print(f"sentence_probs_percentages:\t{str(sentence_probs_percentages)}")
    # sys.exit(1)

    normalized_sentence_probs_percentages = np.array([
        normalize(sentence_probs_percentages[:, 0]) * 100,
        normalize(sentence_probs_percentages[:, 1]) * 100,
        normalize(sentence_probs_percentages[:, 2]) * 100
    ]).T

    print(f"normalized_sentence_probs_percentages:\t{str(normalized_sentence_probs_percentages)}")
    # sys.exit(1)
    # sentence_probs_percentages = normalized_sentence_probs_percentages

    sentence_probs_percentages_list = sentence_probs_percentages.tolist()

    probs_out_dir = "../sentence_probs_percentages/"
    with open(os.path.join(probs_out_dir, f"{args.id}.txt"), "w", encoding="utf-8") as fw_probs:
        fw_probs.write(str(sentence_probs_percentages_list))

    sentence_mmm_values_B = get_mmm(sentence_probs_percentages, "B")
    sentence_mmm_values_I = get_mmm(sentence_probs_percentages, "I")
    sentence_mmm_values_O = get_mmm(sentence_probs_percentages, "O")

    print(f"\nsentence_mmm_values_B:\t{str(sentence_mmm_values_B)}")
    print(f"sentence_mmm_values_I:\t{str(sentence_mmm_values_I)}")
    print(f"sentence_mmm_values_O:\t{str(sentence_mmm_values_O)}")

    B_threshold = calculate_threshold("B", sentence_mmm_values_B["mean"], sentence_mmm_values_B["median"],
                                      sentence_mmm_values_B["min"], sentence_mmm_values_B["max"],
                                      sentence_mmm_values_B["std"])
    I_threshold = calculate_threshold("I", sentence_mmm_values_I["mean"], sentence_mmm_values_I["median"],
                                      sentence_mmm_values_I["min"], sentence_mmm_values_I["max"],
                                      sentence_mmm_values_I["std"])
    O_threshold = calculate_threshold("O", sentence_mmm_values_O["mean"], sentence_mmm_values_O["median"],
                                      sentence_mmm_values_O["min"], sentence_mmm_values_O["max"],
                                      sentence_mmm_values_O["std"])

    print(f"\nB_threshold:\t{str(B_threshold)}")
    print(f"I_threshold:\t{str(I_threshold)}")
    print(f"O_threshold:\t{str(O_threshold)}")

    # sys.exit(1)

    # sentence_segments = custom_probs_to_segments(sentence_probs_percentages, B_threshold, I_threshold, O_threshold)
    # sentence_segments = custom_probs_to_segments_with_o(sentence_probs_percentages, B_threshold, I_threshold,
    #                                                     O_threshold)

    sentence_segments = custom_probs_to_segments_simple(sentence_probs_percentages, 64)

    # sentence_segments = custom_probs_to_segments_simple_2(sentence_probs_percentages)

    # sentence_segments = probs_to_segments(sentence_probs_percentages, B_threshold, O_threshold, I_threshold,
    #                                       threshold_likeliest=False, restart_on_b=True)

    print(f"\nsentence_segments:\t{str(sentence_segments)}")

    original_sentence_segments = deepcopy(sentence_segments)

    ###################################################################################################################

    sentence_segments_postprocessed = postprocess_segments(original_sentence_segments, fps, overall_len, 0.75, 0.75,
                                                           0.)
    print(f"\nsentence_segments_postprocessed:\t{str(sentence_segments_postprocessed)}")

    ###################################################################################################################

    # print(f"\nsentence_segments:\t{str(sentence_segments)}")

    # print(f"sign_segments:\n{str(sign_segments)}\n\n")

    # Save results to various files…

    # Write .eaf files
    # print('Building ELAN files…')
    #
    # tiers_non_postprocessed = {
    #     "SIGN": sign_segments,
    #     "SENTENCE": sentence_segments,
    # }
    #
    # tiers_postprocessed = {
    #     "SIGN": sign_segments,
    #     "SENTENCE": sentence_segments_postprocessed,
    # }
    #
    # create_eaf(args, tiers_postprocessed, fps, boundary_type="milliseconds", postprocessed=True)
    # create_eaf(args, tiers_non_postprocessed, fps, boundary_type="milliseconds", postprocessed=False)
    # create_eaf(args, tiers_postprocessed, fps, boundary_type="frames", postprocessed=True)
    # create_eaf(args, tiers_non_postprocessed, fps, boundary_type="frames", postprocessed=False)

    print(f"\nFinal sentence_segments check:\n{str(sentence_segments)}\n")

    # Write pickle files
    if args.segments_pickle_dir != None:
        print(f"\nWriting pickle-formatted segmentation data structures to {args.segments_pickle_dir}…")
        save_to_pickle(sentence_segments_postprocessed,
                       os.path.join(args.segments_pickle_dir, "POSTPROCESSED", args.id + ".pkl"))
        save_to_pickle(sentence_segments, os.path.join(args.segments_pickle_dir, "NON_POSTPROCESSED", args.id + ".pkl"))

    # Write plaintext files
    if args.segments_plaintext_dir != None:
        print(f"\nWriting plaintext-formatted segmentation data structures to {args.segments_plaintext_dir}…")
        with open(os.path.join(args.segments_plaintext_dir, "POSTPROCESSED", args.id + ".txt"), "w",
                  encoding="utf-8") as fw:
            fw.write(str(sentence_segments_postprocessed))
        with open(os.path.join(args.segments_plaintext_dir, "NON_POSTPROCESSED", args.id + ".txt"), "w",
                  encoding="utf-8") as fw:
            fw.write(str(sentence_segments))


if __name__ == '__main__':
    main()

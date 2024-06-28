import numpy as np
import os
import sys

BIO = {"O": 0, "B": 1, "I": 2}


def io_probs_to_segments(probs, i_threshold):
    segments = []
    i = 0
    while i < len(probs):
        if probs[i, BIO["I"]] > i_threshold:
            end = len(probs) - 1
            for j in range(i + 1, len(probs)):
                if probs[j, BIO["I"]] < i_threshold:
                    end = j - 1
                    break
            segments.append({"start": i, "end": end})
            i = end + 1
        else:
            i += 1

    return segments


def probs_to_segments(probs, b_threshold=50., o_threshold=50., i_threshold=50., threshold_likeliest=False, restart_on_b=True):
    # probs = np.round(np.exp(logits.numpy().squeeze()) * 100)

    np.set_printoptions(threshold=np.inf)

    # print(probs[:, BIO["B"]])

    if np.alltrue(probs[:, BIO["B"]] < b_threshold):
        print(f"b_threshold:\t{str(b_threshold)}")
        print('np.alltrue(probs[:, BIO["B"]] < b_threshold')
        # sys.exit(1)
        return io_probs_to_segments(probs, i_threshold)


    # print(f"probs shape:\t{str(probs.shape)}")
    # print(f"\nprobs:\n{str(probs)}\n")

    # with open(os.path.join("/home/ec2-user/andrew_messaround/vsl_phrase_segmentation/messaround", "example_numpy_tensor_right.txt"), "w", encoding="utf-8") as fw:
    #     print("Saving numpy tensor of probabilities to disk.")
    #     fw.write(str(probs))
    #     print("…Saved.")

    segments = []

    segment = {"start": None, "end": None}
    did_pass_start = False
    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if threshold_likeliest:
            b_threshold = max(i, o)
            o_threshold = max(b, i)

        if segment["start"] is None:
            if b > b_threshold:
                segment["start"] = idx
        else:
            if did_pass_start:
                if (restart_on_b and b > b_threshold) or o > o_threshold:
                    segment["end"] = idx - 1

                    # reset
                    segments.append(segment)
                    segment = {"start": None if o > o_threshold else idx, "end": None}
                    did_pass_start = False
            else:
                if b < b_threshold:
                    did_pass_start = True

    if segment["start"] is not None:
        segment["end"] = len(probs)
        segments.append(segment)

    return segments


def find_true_spans(data):
    spans = []
    start = None

    for i, value in enumerate(data):
        if value:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append([start, i - 1])
                start = None

    # Handle case where the last value is True
    if start is not None:
        spans.append([start, len(data) - 1])

    return spans

# def convert_spans_to_segments(spans):
#     segments = []
#     for span in spans:
#         segment = {"start": span[0], "end": span[1]}
#         segments.append(segment)
#     return segments

def convert_spans_to_segments(spans):
    segments = []
    for span in spans:
        segment = {"start": span[0], "end": span[1]}
        segments.append(segment)
    return segments


def custom_probs_to_segments(probs, b_threshold, i_threshold, o_threshold):
    frames_T_F = []
    # print(f"\nlen(probs):\t{str(len(probs))}")

    # First, determine if each frame meets the segment inclusion criteria
    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        # Append True or False to frames_T_F based on the thresholds
        if b >= b_threshold:
            frames_T_F.append(True)
        elif i >= i_threshold:
            frames_T_F.append(True)
        else:
            frames_T_F.append(False)

    # print(f"\nlen(frames_T_F): {len(frames_T_F)}")
    # print(f"frames_T_F:\n{frames_T_F}")

    spans = find_true_spans(frames_T_F)
    # print(f"\nspans:\t{str(spans)}")
    segments = convert_spans_to_segments(spans)
    # print(f"\nsegments:\t{str(segments)}")

    return segments


def custom_probs_to_segments_with_o(probs, b_threshold, i_threshold, o_threshold):
    frames_T_F = []
    # print(f"\nlen(probs):\t{str(len(probs))}")

    # Determine if each frame meets the segment inclusion criteria
    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if o >= o_threshold:
            frames_T_F.append(False)
        else:
            if b >= b_threshold:
                frames_T_F.append(True)
            elif i >= i_threshold:
                frames_T_F.append(True)
            else:
                frames_T_F.append(False)

    # print(f"\nlen(frames_T_F): {len(frames_T_F)}")

    spans = find_true_spans(frames_T_F)
    segments = convert_spans_to_segments(spans)

    return segments


def custom_probs_to_segments_simple(probs, o_threshold):
    frames_T_F = []

    # print(f"BIO['O']:\t{str(BIO['O'])}")

    # print(f"probs (within probs_to_segments.custom_probs_to_segments_simple()):\n{probs}")
    # print(f"\nlen(probs):\t{str(len(probs))}")

    for idx in range(len(probs)):
        o = float(probs[idx, BIO["O"]])

        # print(f"Frame {idx}: o = {o}, o >= o_threshold = {o >= o_threshold}")

        if o >= o_threshold:
            frames_T_F.append(False)
        else:
            frames_T_F.append(True)

    # print(f"frames_T_F:\n{frames_T_F}")

    spans = find_true_spans(frames_T_F)
    # print(f"spans:\n{spans}")
    segments = convert_spans_to_segments(spans)
    # print(f"segments:\n{segments}")

    return segments


def custom_probs_to_segments_simple_2(probs):
    frames_T_F = []

    # print(f"probs:\n{probs}")
    # print(f"\nlen(probs):\t{str(len(probs))}")

    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if b > o or i > o:
            frames_T_F.append(True)
        else:
            frames_T_F.append(False)

    # print(f"frames_T_F:\n{frames_T_F}")

    spans = find_true_spans(frames_T_F)
    segments = convert_spans_to_segments(spans)

    return segments


def custom_probs_to_segments_simple_i(probs, i_threshold):
    frames_T_F = []

    # print(f"BIO['O']:\t{str(BIO['O'])}")

    # print(f"probs:\n{probs}")
    # print(f"\nlen(probs):\t{str(len(probs))}")

    for idx in range(len(probs)):
        i = float(probs[idx, BIO["I"]])

        # print(f"Frame {idx}: o = {o}, o >= o_threshold = {o >= o_threshold}")

        if i >= i_threshold:
            frames_T_F.append(True)
        else:
            frames_T_F.append(False)

    # print(f"frames_T_F:\n{frames_T_F}")

    spans = find_true_spans(frames_T_F)
    # print(f"spans:\n{spans}")
    segments = convert_spans_to_segments(spans)
    # print(f"segments:\n{segments}")

    return segments



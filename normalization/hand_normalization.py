import pandas as pd

HAND_IDENTIFIERS = [
    'wrist',
    'thumb_cmc',
    'thumb_mcp', 
    'thumb_ip',
    'thumb_tip',
    'index_mcp', 
    'index_pip', 
    'index_dip', 
    'index_tip', 
    'middle_mcp', 
    'middle_pip',
    'middle_dip', 
    'middle_tip', 
    'ring_mcp',
    'ring_pip', 
    'ring_dip',
    'ring_tip', 
    'pinky_mcp', 
    'pinky_pip', 
    'pinky_dip', 
    'pinky_tip', 
    'wrist',
    'thumb_cmc',
    'thumb_mcp', 
    'thumb_ip',
    'thumb_tip',
    'index_mcp', 
    'index_pip', 
    'index_dip', 
    'index_tip', 
    'middle_mcp', 
    'middle_pip',
    'middle_dip', 
    'middle_tip', 
    'ring_mcp',
    'ring_pip', 
    'ring_dip',
    'ring_tip', 
    'pinky_mcp', 
    'pinky_pip', 
    'pinky_dip', 
    'pinky_tip', 
]

def normalize_hands_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the hands position data using the Bohacek-normalization algorithm.

    :param df: pd.DataFrame to be normalized
    :return: pd.DataFrame with normalized values for hand pose
    """

    empty = 0
    df.columns = [item.replace("L_", "_0_").replace("R_", "_1_") for item in list(df.columns)]

    normalized_df = pd.DataFrame(columns=df.columns)

    hand_landmarks = {"X": {0: [], 1: []}, "Y": {0: [], 1: []}}

    # Determine how many hands are present in the dataset
    range_hand_size = 1
    if "wrist_1_X" in df.columns:
        range_hand_size = 2

    # Construct the relevant identifiers
    for identifier in HAND_IDENTIFIERS:
        for hand_index in range(range_hand_size):
            hand_landmarks["X"][hand_index].append(identifier + "_" + str(hand_index) + "_X")
            hand_landmarks["Y"][hand_index].append(identifier + "_" + str(hand_index) + "_Y")

    # Iterate over all of the records in the dataset
    for index, row in df.iterrows():
        # Treat each hand individually
        for hand_index in range(range_hand_size):

            sequence_size = len(row["wrist_" + str(hand_index) + "_X"])

            # Treat each element of the sequence (analyzed frame) individually
            for sequence_index in range(sequence_size):

                # Retrieve all of the X and Y values of the current frame
                landmarks_x_values = [row[key][sequence_index] for key in hand_landmarks["X"][hand_index] if row[key][sequence_index] != 0]
                landmarks_y_values = [row[key][sequence_index] for key in hand_landmarks["Y"][hand_index] if row[key][sequence_index] != 0]

                # Prevent from even starting the analysis if some necessary elements are not present
                if not landmarks_x_values or not landmarks_y_values:
                    empty += 1
                    continue
                

                # Calculate the deltas
                width, height = max(landmarks_x_values) - min(landmarks_x_values), max(landmarks_y_values) - min(
                    landmarks_y_values)
                if width > height:
                    delta_x = 0.1 * width
                    delta_y = delta_x + ((width - height) / 2)
                else:
                    delta_y = 0.1 * height
                    delta_x = delta_y + ((height - width) / 2)

                # Set the starting and ending point of the normalization bounding box
                starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
                ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

                # Normalize individual landmarks and save the results
                for identifier in HAND_IDENTIFIERS:
                    key = identifier + "_" + str(hand_index) + "_"

                    # Prevent from trying to normalize incorrectly captured points
                    if row[key + "X"][sequence_index] == 0 or (ending_point[0] - starting_point[0]) == 0 or (starting_point[1] - ending_point[1]) == 0:
                        continue

                    normalized_x = (row[key + "X"][sequence_index] - starting_point[0]) / (ending_point[0] - starting_point[0])
                    normalized_y = (row[key + "Y"][sequence_index] - ending_point[1]) / (starting_point[1] - ending_point[1])

                    row[key + "X"][sequence_index] = normalized_x
                    row[key + "Y"][sequence_index] = normalized_y

        normalized_df = pd.concat([normalized_df, pd.DataFrame([row])], ignore_index=True)
    
    return normalized_df


def normalize_single_dict(row: dict): 
    """
    Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
    the definition from our paper.

    :param row: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                that particular joints coordinates # joint: [frames,(X,Y)] shape array
    :return: Dictionary with normalized skeletal data (following the same schema as input data)
    """

    hand_landmarks = {'R': [], 'L': []}
    hands = ['R', 'L']

    # Construct the relevant identifiers
    for identifier in HAND_IDENTIFIERS:
        ### for hand_index in range(range_hand_size):
        for hand_index in hands:
            hand_landmarks[hand_index].append(identifier + hand_index)

    # Treat each hand individually
    ### for hand_index in range(range_hand_size):
    for hand_index in hands:

        sequence_size = len(row["wrist" + hand_index])

        # Treat each element of the sequence (analyzed frame) individually
        for sequence_index in range(sequence_size):

            # Retrieve all of the X and Y values of the current frame
            landmarks_x_values = [row[key][sequence_index][0] for key in hand_landmarks[hand_index] if
                                row[key][sequence_index][0] != 0]
            landmarks_y_values = [row[key][sequence_index][1] for key in hand_landmarks[hand_index] if
                                row[key][sequence_index][1] != 0]

            # Prevent from even starting the analysis if some necessary elements are not present
            if not landmarks_x_values or not landmarks_y_values:
                continue

            # Calculate the deltas
            width, height = max(landmarks_x_values) - min(landmarks_x_values), max(landmarks_y_values) - min(
                landmarks_y_values)
            if width > height:
                delta_x = 0.1 * width
                delta_y = delta_x + ((width - height) / 2)
            else:
                delta_y = 0.1 * height
                delta_x = delta_y + ((height - width) / 2)

            # Set the starting and ending point of the normalization bounding box
            starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
            ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

            # Normalize individual landmarks and save the results
            for identifier in HAND_IDENTIFIERS:
                key = identifier + hand_index

                # Prevent from trying to normalize incorrectly captured points
                if row[key][sequence_index][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                        starting_point[1] - ending_point[1]) == 0:
                    continue

                normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
                normalized_y = (row[key][sequence_index][1] - starting_point[1]) / (ending_point[1] - starting_point[1])

                row[key][sequence_index] = list(row[key][sequence_index])

                row[key][sequence_index][0] = normalized_x
                row[key][sequence_index][1] = normalized_y

    return row


if __name__ == "__main__":
    pass

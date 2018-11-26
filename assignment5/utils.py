import bz2
import matplotlib.pyplot as plt
import random
import tensorflow as tf


def load_lines(file_name, fields, delimiter, n=0, compression=None, no_id=False, skip_first=False):
    lines = {}

    if compression is "bz2":
        open_fn = bz2.open
        open_mode = "rt"
    else:
        open_fn = open
        open_mode = "r"

    read = 0
    with open_fn(file_name, open_mode) as f:
        for line in f:
            if skip_first:
                skip_first = False
                continue
                
            # print(line)
                
            values = line.strip().split(delimiter)
            lineObj = {}
            try:
                # Extract fields
                for i, field in enumerate(fields):
                    if field == "lineID":
                        lineObj[field] = "ID" + values[i]
                    else:
                        lineObj[field] = values[i]

                # Add lineID if not found in original file
                if no_id:
                    lineObj["lineID"] = "ID{}".format(read)

                lines[lineObj['lineID']] = lineObj

                read += 1
            except IndexError:
                continue
                
            if (n > 0) and (read == n):
                break

    return lines


def extract_instances_quora(lines, n=0):
    src = []
    tgt = []
    labels = []

    for line_id, line_obj in lines.items():
        if n > 0 and len(src) == n:
            break

        src.append(line_obj["source"])
        tgt.append(line_obj["target"])
        labels.append(line_obj["label"])

    return src, tgt, labels


def extract_instances(lines, mode, n=0, label_neg=0):
    src = []
    tgt = []
    labels = []

    for line_id, line_obj in lines[mode].items():
        if mode == "dev" and float(line_obj["annotation"]) == 2.5:
            continue

        if n > 0 and len(src) == n:
            break

        src.append(line_obj["source"])
        tgt.append(line_obj["target"])

        if mode == "train":
            labels.append(1)
        else:
            label = 1 if float(line_obj["annotation"]) > 2.5 else label_neg
            labels.append(label)

    return src, tgt, labels


def make_negatives(lines, n=0, label_neg=0):
    src, tgt, _ = extract_instances(lines, "train")
    random.shuffle(src)

    labels = [label_neg]*len(src)

    if n > 0:
        src = src[:n]
        tgt = tgt[:n]
        labels = labels[:n]
    return src, tgt, labels


def get_instances(n_pos=0, n_neg=-1, part="train"):
    n_neg = n_pos if n_neg == -1 else n_neg

    filepaths = {
        "train": "data/opusparcus/en/train/en-train.txt.bz2",
        "dev": "data/opusparcus/en/dev/en-dev.txt",
        "test": "data/opusparcus/en/test/en-test.txt"
    }

    fields_train = "lineID source target score numAlign pivotLangs editDist".split(" ")
    fields_dev = "lineID source target annotation".split(" ")

    lines = {
        "train": None,
        "dev": None,
        "test": None,
    }

    compression = "bz2" if part == "train" else None
    fields = fields_train if part == "train" else fields_dev
    lines[part] = load_lines(
        filepaths[part], fields, delimiter="\t",
        n=n_pos, compression=compression
    )

    if part == "train":
        src_pos_string, tgt_pos_string, labels_pos = \
            extract_instances(lines, part, n=n_pos)
        src_neg_string, tgt_neg_string, labels_neg = \
            make_negatives(lines, n=n_neg)

        src_all_string = src_pos_string + src_neg_string
        tgt_all_string = tgt_pos_string + tgt_neg_string
        labels = labels_pos + labels_neg
    else:
        src_all_string, tgt_all_string, labels = \
            extract_instances(lines, part)

    return src_all_string, tgt_all_string, labels


def plot_training(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    


def contrastive_loss(y_true, y_pred):
    """Calculate a simple contrastive loss based on cosine distance."""

    # Turn cosine similarities into cosine distances
    distances = 1. - y_pred
    # Make sure labels are float
    labels_f = tf.to_float(y_true)

    # Cost when label == 0
    term_1 = (1. - labels_f)*tf.square(tf.maximum(0., 0.8 - distances))
    # Cost when label == 1
    term_2 = labels_f*distances

    # Return the sum of the two terms (the full loss)
    return term_1 + term_2


def load_embs():
    embeddings_index = {}
    with open("data/embeddings/") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
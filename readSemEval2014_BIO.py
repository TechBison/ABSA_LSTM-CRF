from nltk.tokenize import word_tokenize
import numpy as np
import xml.etree.ElementTree as ET


def insert_in_array(original_array, sub_array, index):
    original_array = np.insert(original_array, index + 1, sub_array)
    original_array = np.delete(original_array, index)
    return original_array


def get_BIO(xml_file):
    X = []
    Y = []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for k, sentence in enumerate(root.findall("./sentence")):
        review = sentence.find('text').text
        review = review.lower()
        review_list = list(review)

        at_tuple = []

        i = 0
        for aspectTerm in sentence.findall("./aspectTerms/aspectTerm"):
            aspect_term = aspectTerm.get('term')
            aspect_term = aspect_term.lower()
            aspect_polarity = aspectTerm.get('polarity')
            from_ = int(aspectTerm.get('from'))
            to_ = int(aspectTerm.get('to'))
            term_length = len(aspect_term)

            review_list[from_:to_] = np.repeat(str(i), term_length)

            AP = ''
            if aspect_polarity:
                AP = aspect_polarity[0].upper()

            at_tuple.append((str(i), term_length, aspect_term, AP))

            i += 1

        # convert to lower case
        tokens = word_tokenize(''.join(review_list))
        tokens = np.array(tokens)
        labels = np.full(np.shape(tokens), 'O')

        for t in at_tuple:
            rep_word = ''.join(np.repeat(t[0], t[1]))
            act_word = t[2]
            # polarity = t[3]

            aspect_tokens = word_tokenize(act_word)
            aspect_labels = np.full(np.shape(aspect_tokens), 'I')
            aspect_labels[0] = 'B'

            ind = np.where(tokens == rep_word)
            if ind[0].size > 0:
                tokens = insert_in_array(tokens, aspect_tokens, ind[0])
                labels = insert_in_array(labels, aspect_labels, ind[0])

        # print(*tokens)
        X.append(tokens)
        Y.append(labels)
    return X, Y


def get_Categories(xml_file='Restaurants_Train_v2.xml'):
    X = []
    Y = []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for k, sentence in enumerate(root.findall("./sentence")):
        review = sentence.find('text').text
        review = review.lower()
        review_list = list(review)

        labels = []
        for aspectTerm in sentence.findall("./aspectCategories/aspectCategory"):
            category = aspectTerm.get('category')
            category = category.lower()
            aspect_polarity = aspectTerm.get('polarity')
            labels.append(category)


        # convert to lower case
        tokens = word_tokenize(''.join(review_list))
        tokens = np.array(tokens)

        # print(*tokens)
        X.append(tokens)
        Y.append(labels)
    return X, Y


def get_aspect_terms(labels, input_tokens):
    aspect_list = []
    asp = None
    for ind, val in enumerate(labels):

        ########################
        # bio_index = {'O': 0, 'B': 1, 'I': 2}
        ########################

        next_ind = ind + 1

        if asp is None:
            asp = '<UNK>'
        if next_ind < len(labels):
            if val == 'B' and labels[next_ind] == 'O':
                aspect_list.append(input_tokens[ind])
            elif val == 'B' and labels[next_ind] == 'I':
                asp = input_tokens[ind]
            elif val == 'I':
                asp = asp + ' ' + input_tokens[ind]
                if not labels[next_ind] == 'I':
                    aspect_list.append(asp)
        else:
            if val == 'B':
                aspect_list.append(input_tokens[ind])
            elif val == 'I':
                asp = asp + ' ' + input_tokens[ind]
                aspect_list.append(asp)

    return aspect_list


if __name__ == '__main__':
    file = '../Data/Laptop_Train_v2.xml'
    X, Y  = get_BIO(xml_file=file)

    for i in range(len(X)):
        print(X[i])
        print(Y[i])

        print('\n')



#     for i in zip(tokens,labels):
#         print(i)
#     print('\n')

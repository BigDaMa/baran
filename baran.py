########################################
# Raha 2
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import re
import io
import json
import html
import pickle
import unicodedata
import difflib
import libarchive.public
import bs4
import bz2
import numpy
import mwparserfromhell
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import dataset
########################################


########################################
class Raha2:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor sets up.
        """
        self.DATASETS_FOLDER = "datasets"
        self.RESULTS_FOLDER = "results"
        self.WIKIPEDIA_DUMPS_FOLDER = "../wikipedia-data-sample"
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.CLASSIFICATION_MODEL = "ABC"   # ["ABC", "DTC", "GBC", "GNB", "KNC" ,"SGDC", "SVC"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.ONLINE_PHASE = False
        self.RUN_COUNT = 1
        self.LABELING_BUDGET = 20
        self.MIN_FD_DEGREE = 0.95
        self.MIN_CORRECTION_PROBABILITY = 0.01
        self.MAX_VALUE_LENGTH = 50
        self.CONTEXT_WINDOW_SIZE = 5

    @staticmethod
    def wiki_text_segmenter(text):
        """
        This method is a custom segmenter that takes a wikipedia text and segments it.
        """
        def recursive_parser(node):
            if not node:
                pass
            elif isinstance(node, str):
                segments_list.append(node)
            elif isinstance(node, mwparserfromhell.nodes.text.Text):
                segments_list.append(node.value)
            elif isinstance(node, mwparserfromhell.wikicode.Wikicode):
                for n in node.nodes:
                    if isinstance(n, str) or isinstance(n, mwparserfromhell.nodes.text.Text) or not n:
                        recursive_parser(n)
                    elif isinstance(n, mwparserfromhell.nodes.heading.Heading):
                        recursive_parser(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.tag.Tag):
                        recursive_parser(n.contents)
                    elif isinstance(n, mwparserfromhell.nodes.wikilink.Wikilink):
                        if n.text:
                            recursive_parser(n.text)
                        else:
                            recursive_parser(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.external_link.ExternalLink):
                        # recursive_parser(n.url)
                        recursive_parser(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.template.Template):
                        recursive_parser(n.name)
                        for p in n.params:
                            # recursive_parser(p.name)
                            recursive_parser(p.value)
                    elif isinstance(n, mwparserfromhell.nodes.html_entity.HTMLEntity):
                        segments_list.append(n.normalize())
                    elif isinstance(n, mwparserfromhell.nodes.comment.Comment) or \
                            isinstance(n, mwparserfromhell.nodes.argument.Argument):
                        pass
                    else:
                        print("Second layer unknown node:", type(n), n)
            else:
                print("First layer unknown node:", type(node), node)

        segments_list = []
        wiki_code = mwparserfromhell.parse(text)
        recursive_parser(wiki_code)
        return segments_list

    def revision_extractor(self):
        """
        This method extracts the revision data from the wikipedia page revisions.
        """
        rd_folder_path = os.path.join(self.RESULTS_FOLDER, "revision-data")
        if not os.path.exists(rd_folder_path):
            os.mkdir(rd_folder_path)
        compressed_dumps_list = os.listdir(self.WIKIPEDIA_DUMPS_FOLDER)
        page_counter = 0
        for file_name in compressed_dumps_list:
            compressed_dump_file_path = os.path.join(self.WIKIPEDIA_DUMPS_FOLDER, file_name)
            if not compressed_dump_file_path.endswith(".7z"):
                continue
            dump_file_name, _ = os.path.splitext(os.path.basename(compressed_dump_file_path))
            rdd_folder_path = os.path.join(rd_folder_path, dump_file_name)
            if not os.path.exists(rdd_folder_path):
                os.mkdir(rdd_folder_path)
            else:
                continue
            decompressed_dump_file_path = os.path.splitext(compressed_dump_file_path)[0]
            with libarchive.public.file_reader(compressed_dump_file_path) as e:
                for entry in e:
                    with open(decompressed_dump_file_path, "wb") as f:
                        for block in entry.get_blocks():
                            f.write(block)
            # compressed_dump_file = io.open(compressed_dump_file_path, "rb")
            # compressed_dump = py7zlib.Archive7z(compressed_dump_file)
            # decompressed_dump_file = io.open(decompressed_dump_file_path, "w")
            # for f in compressed_dump.getmembers():
            #    data = f.read().decode("utf-8")
            #    decompressed_dump_file.write(data)
            # decompressed_dump_file.close()
            decompressed_dump_file = io.open(decompressed_dump_file_path, "r", encoding="utf-8")
            page_text = ""
            for i, line in enumerate(decompressed_dump_file):
                line = line.strip()
                if line == "<page>":
                    page_text = ""
                page_text += "\n" + line
                if line == "</page>":
                    revisions_list = []
                    page_tree = bs4.BeautifulSoup(page_text, "html.parser")
                    previous_text = ""
                    for revision_tag in page_tree.find_all("revision"):
                        revision_text = revision_tag.find_all("text")[0].text
                        if previous_text:
                            a = [t for t in self.wiki_text_segmenter(previous_text) if t]
                            b = [t for t in self.wiki_text_segmenter(revision_text) if t]
                            s = difflib.SequenceMatcher(None, a, b)
                            for tag, i1, i2, j1, j2 in s.get_opcodes():
                                if tag == "equal":
                                    continue
                                revisions_list.append({
                                    "old_value": a[i1:i2],
                                    "new_value": b[j1:j2],
                                    "left_context": a[i1 - self.CONTEXT_WINDOW_SIZE:i1],
                                    "right_context": a[i2:i2 + self.CONTEXT_WINDOW_SIZE]
                                })
                        previous_text = revision_text
                    if revisions_list:
                        page_counter += 1
                        if page_counter % 100 == 0:
                            for entry in revisions_list:
                                print("----------Page Counter:---------\n", page_counter,
                                      "\n----------Old Value:---------\n", entry["old_value"],
                                      "\n----------New Value:---------\n", entry["new_value"],
                                      "\n----------Left Context:---------\n", entry["left_context"],
                                      "\n----------Right Context:---------\n", entry["right_context"],
                                      "\n==============================")
                        json.dump(revisions_list, open(os.path.join(rdd_folder_path, page_tree.id.text + ".json"), "w"))
            decompressed_dump_file.close()
            os.remove(decompressed_dump_file_path)

    @staticmethod
    def value_normalizer(value):
        """
        This method normalizes a value.
        """
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
        return value

    @staticmethod
    def value_encoder(value, encoding):
        """
        This method represents a value with a specified encoding method.
        """
        if encoding == "identity":
            return json.dumps(list(value))
        if encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])

    @staticmethod
    def add_to_model_helper(model, key, value):
        """
        This methods adds a key-value into a model dictionary.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    def value_based_models_updater(self, models, ud):
        """
        This method updates the value-based error corrector models.
        """
        # TODO: building "shifter" model for dd.mm.yy -> mm.dd.yy
        if self.ONLINE_PHASE or (ud["new_value"] and len(ud["new_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] and len(ud["old_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] != ud["new_value"]):
            remover_transformation = {}
            adder_transformation = {}
            replacer_transformation = {}
            s = difflib.SequenceMatcher(None, ud["old_value"], ud["new_value"])
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                index_range = json.dumps([i1, i2])
                if tag == "delete":
                    remover_transformation[index_range] = ""
                if tag == "insert":
                    adder_transformation[index_range] = ud["new_value"][j1:j2]
                if tag == "replace":
                    replacer_transformation[index_range] = ud["new_value"][j1:j2]
            for encoding in self.VALUE_ENCODINGS:
                encoded_old_value = self.value_encoder(ud["old_value"], encoding)
                if remover_transformation:
                    self.add_to_model_helper(models[0], encoded_old_value, json.dumps(remover_transformation))
                if adder_transformation:
                    self.add_to_model_helper(models[1], encoded_old_value, json.dumps(adder_transformation))
                if replacer_transformation:
                    self.add_to_model_helper(models[2], encoded_old_value, json.dumps(replacer_transformation))
                self.add_to_model_helper(models[3], encoded_old_value, ud["new_value"])

    def context_based_models_updater(self, models, ud):
        """
        This method updates the context-based error corrector models.
        """
        for j, cv in enumerate(ud["context"]):
            if cv != self.IGNORE_SIGN:
                self.add_to_model_helper(models[j][ud["column"]], cv, ud["new_value"])

    def domain_based_model_updater(self, model, ud):
        """
        This method updates the domain-based error corrector model.
        """
        self.add_to_model_helper(model, ud["column"], ud["new_value"])

    def value_based_models_pretrainer(self):
        """
        This method pretrains value-based error corrector models.
        """
        cm_folder_path = os.path.join(self.RESULTS_FOLDER, "corrector-models")
        if not os.path.exists(cm_folder_path):
            os.mkdir(cm_folder_path)
        models = [{}, {}, {}, {}]
        rd_folder_path = os.path.join(self.RESULTS_FOLDER, "revision-data")
        page_counter = 0
        for folder in os.listdir(rd_folder_path):
            for f in os.listdir(os.path.join(rd_folder_path, folder)):
                page_counter += 1
                if page_counter % 100 == 0:
                    print(page_counter, "pages processed.")
                revision_list = json.load(io.open(os.path.join(rd_folder_path, folder, f), encoding="utf-8"))
                for r in revision_list:
                    update_dictionary = {
                        "old_value": self.value_normalizer("".join(r["old_value"])),
                        "new_value": self.value_normalizer("".join(r["new_value"]))
                    }
                    self.value_based_models_updater(models, update_dictionary)
        pruned_models = []
        for model in models:
            new_model = {}
            pruning_threshold = 0.0
            ctr = 0
            for k in model:
                for v in model[k]:
                    pruning_threshold += model[k][v]
                    ctr += 1
            pruning_threshold /= ctr
            print("Pruning Threshold =", pruning_threshold)
            for k in model:
                for v in model[k]:
                    if model[k][v] >= pruning_threshold:
                        self.add_to_model_helper(new_model, k, v)
                        new_model[k][v] = model[k][v]
            pruned_models.append(new_model)
        pickle.dump(pruned_models, bz2.BZ2File(os.path.join(cm_folder_path, "value_models.dictionary"), "wb"))

    def value_based_corrector(self, models, ed):
        """
        This method takes the value-based models and an error dictionary to generate potential value-based corrections.
        """
        results_list = []
        for m, model_name in enumerate(["remover", "adder", "replacer", "swapper"]):
            model = models[m]
            for encoding in self.VALUE_ENCODINGS:
                results_dictionary = {}
                encoded_value_string = self.value_encoder(ed["old_value"], encoding)
                if encoded_value_string in model:
                    sum_scores = sum(model[encoded_value_string].values())
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            index_character_dictionary = {i: c for i, c in enumerate(ed["old_value"])}
                            transformation = json.loads(transformation_string)
                            for change_range_string in transformation:
                                change_range = json.loads(change_range_string)
                                if model_name in ["remover", "replacer"]:
                                    for i in range(change_range[0], change_range[1]):
                                        index_character_dictionary[i] = ""
                                if model_name in ["adder", "replacer"]:
                                    ov = "" if change_range[0] not in index_character_dictionary else \
                                        index_character_dictionary[change_range[0]]
                                    index_character_dictionary[change_range[0]] = transformation[change_range_string] + ov
                            new_value = ""
                            for i in range(len(index_character_dictionary)):
                                new_value += index_character_dictionary[i]
                            p = model[encoded_value_string][transformation_string] / sum_scores
                            if p >= self.MIN_CORRECTION_PROBABILITY:
                                results_dictionary[new_value] = p
                    if model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            p = model[encoded_value_string][new_value] / sum_scores
                            if p >= self.MIN_CORRECTION_PROBABILITY:
                                results_dictionary[new_value] = p
                results_list.append(results_dictionary)
        return results_list

    def context_based_corrector(self, models, ed):
        """
        This method takes the context-based models and an error dictionary to generate potential context-based corrections.
        """
        results_list = []
        for j, cv in enumerate(ed["context"]):
            results_dictionary = {}
            if j != ed["column"] and cv in models[j][ed["column"]]:
                sum_scores = sum(models[j][ed["column"]][cv].values())
                for new_value in models[j][ed["column"]][cv]:
                    p = models[j][ed["column"]][cv][new_value] / sum_scores
                    if p >= self.MIN_CORRECTION_PROBABILITY:
                        results_dictionary[new_value] = p
            results_list.append(results_dictionary)
        return results_list

    def domain_based_corrector(self, model, ed):
        """
        This method takes a domain-based model and an error dictionary to generate potential domain-based corrections.
        """
        results_dictionary = {}
        sum_scores = sum(model[ed["column"]].values())
        for new_value in model[ed["column"]]:
            p = model[ed["column"]][new_value] / sum_scores
            if p >= self.MIN_CORRECTION_PROBABILITY:
                results_dictionary[new_value] = p
        return results_dictionary

    def error_corrector(self, d, all_errors):
        """
        This method generates the possible corrections for each data error and learns to predict the right ones.
        """
        self.ONLINE_PHASE = True
        data_errors_per_column = {}
        for cell in all_errors:
            self.add_to_model_helper(data_errors_per_column, cell[1], cell)
        cm_folder_path = os.path.join(self.RESULTS_FOLDER, "corrector-models")
        value_models = pickle.load(bz2.BZ2File(os.path.join(cm_folder_path, "value_models.dictionary"), "rb"))
        context_models = {j: {jj: {} for jj in range(d.dataframe.shape[1])} for j in range(d.dataframe.shape[1])}
        domain_model = {}
        for i in range(d.dataframe.shape[0]):
            if i % 1000 == 0:
                print(i)
            for j in range(d.dataframe.shape[1]):
                # TODO
                # d.dataframe.iloc[i, j] = self.value_normalizer(d.dataframe.iloc[i, j])
                if (i, j) not in all_errors:
                    update_dictionary = {
                        "column": j,
                        "new_value": d.dataframe.iloc[(i, j)],
                        "context": [cv if (j != cj and (i, cj) not in all_errors)
                                    else self.IGNORE_SIGN for cj, cv in enumerate(d.dataframe.iloc[i, :])]
                    }
                    self.context_based_models_updater(context_models, update_dictionary)
                    self.domain_based_model_updater(domain_model, update_dictionary)
        fd_degree = numpy.zeros((d.dataframe.shape[1], d.dataframe.shape[1]))
        for j in range(d.dataframe.shape[1]):
            for jj in range(d.dataframe.shape[1]):
                if j != jj:
                    for v in context_models[j][jj]:
                        if len(context_models[j][jj][v]) <= 1:
                            fd_degree[j, jj] += 1
                    if len(context_models[j][jj]) > 0:
                        fd_degree[j, jj] /= len(context_models[j][jj])
                    if fd_degree[j, jj] < self.MIN_FD_DEGREE:
                        context_models[j][jj] = {}
                    else:
                        print(j, jj)
        sampling_range = range(1, self.LABELING_BUDGET + 1)
        aggregate_results = {s: numpy.empty((0, 3)) for s in sampling_range}
        for r in range(self.RUN_COUNT):
            print("Run {}...".format(r))
            labeled_tuples = {}
            correction_dictionary = {}
            while len(labeled_tuples) < self.LABELING_BUDGET:
                remaining_data_errors_per_column = numpy.zeros(d.dataframe.shape[1])
                for cell in all_errors:
                    if cell not in correction_dictionary:
                        remaining_data_errors_per_column[cell[1]] += 1.0
                tuple_score = numpy.zeros(d.dataframe.shape[0])
                for cell in all_errors:
                    if cell not in correction_dictionary:
                        tuple_score[cell[0]] += remaining_data_errors_per_column[cell[1]]
                if sum(tuple_score) == 0:
                    si = numpy.random.choice(range(len(tuple_score)))
                else:
                    # sum_tuple_score = sum(tuple_score)
                    # p_tuple_score = tuple_score / sum_tuple_score
                    # si = numpy.random.choice(numpy.arange(len(tuple_score)), 1, p=p_tuple_score)[0]
                    si = numpy.argmax(tuple_score)
                    # si = numpy.random.choice(list(xxx.keys()))
                labeled_tuples[si] = 1
                for j in range(d.dataframe.shape[1]):
                    cell = (si, j)
                    if cell in all_errors:
                        dirty_value = d.dataframe.iloc[cell]
                        clean_value = d.clean_dataframe.iloc[cell]
                        correction_dictionary[cell] = clean_value
                        update_dictionary = {
                            "column": cell[1],
                            "old_value": dirty_value,
                            "new_value": clean_value,
                        }
                        self.value_based_models_updater(value_models, update_dictionary)
                        self.domain_based_model_updater(domain_model, update_dictionary)
                temp_cleaned_row = [correction_dictionary[(si, j)] if (si, j) in all_errors else v
                                    for j, v in enumerate(d.dataframe.iloc[si, :])]
                for j in range(d.dataframe.shape[1]):
                    update_dictionary = {
                        "column": j,
                        "new_value": temp_cleaned_row[j]
                    }
                    if (si, j) in all_errors:
                        update_dictionary["context"] = [cv if j != cj and fd_degree[cj, j] >= self.MIN_FD_DEGREE
                                                        else self.IGNORE_SIGN for cj, cv in enumerate(temp_cleaned_row)]
                    else:
                        update_dictionary["context"] = [cv if j != cj and fd_degree[cj, j] >= self.MIN_FD_DEGREE and (si, cj) in all_errors
                                                        else self.IGNORE_SIGN for cj, cv in enumerate(temp_cleaned_row)]
                    self.context_based_models_updater(context_models, update_dictionary)
                for j in data_errors_per_column:
                    x_train = []
                    y_train = []
                    x_test = []
                    test_cell_value_list = []
                    error_dictionary = {"column": j}
                    domain_corrections = [self.domain_based_corrector(domain_model, error_dictionary)]
                    for k, cell in enumerate(data_errors_per_column[j]):
                        if self.MIN_CORRECTION_PROBABILITY > 0.0 and cell in correction_dictionary and cell[0] not in labeled_tuples:
                            continue
                        error_dictionary["old_value"] = d.dataframe.iloc[cell]
                        error_dictionary["context"] = list(d.dataframe.iloc[cell[0], :])
                        value_corrections = self.value_based_corrector(value_models, error_dictionary)
                        context_corrections = self.context_based_corrector(context_models, error_dictionary)
                        models_corrections = value_corrections + context_corrections + domain_corrections
                        fv = {}
                        for i, model in enumerate(models_corrections):
                            for v in model:
                                if v not in fv:
                                    fv[v] = numpy.zeros(len(models_corrections))
                                fv[v][i] = model[v]
                        for v in fv:
                            if cell[0] in labeled_tuples:
                                x_train.append(fv[v])
                                y_train.append(int(v == correction_dictionary[cell]))
                            else:
                                x_test.append(fv[v])
                                test_cell_value_list.append([cell, v])
                        if k % 1000 == 0:
                            print(j, "|", k, "/", len(data_errors_per_column[j]), "|", len(fv))
                    if self.CLASSIFICATION_MODEL == "ABC":
                        classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                    if self.CLASSIFICATION_MODEL == "DTC":
                        classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
                    if self.CLASSIFICATION_MODEL == "GBC":
                        classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                    if self.CLASSIFICATION_MODEL == "GNB":
                        classification_model = sklearn.naive_bayes.GaussianNB()
                    if self.CLASSIFICATION_MODEL == "KNC":
                        classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
                    if self.CLASSIFICATION_MODEL == "SGDC":
                        classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                    if self.CLASSIFICATION_MODEL == "SVC":
                        classification_model = sklearn.svm.SVC(kernel="sigmoid")
                    if x_train and x_test:
                        if sum(y_train) == 0:
                            predicted_labels = [0] * len(x_test)
                        elif sum(y_train) == len(y_train):
                            predicted_labels = [1] * len(x_test)
                        else:
                            classification_model.fit(x_train, y_train)
                            predicted_labels = classification_model.predict(x_test)
                        # predicted_probabilities = classification_model.predict_proba(x_test)
                        # correction_confidence = {}
                        for index, pl in enumerate(predicted_labels):
                            cell, pc = test_cell_value_list[index]
                            # confidence = predicted_probabilities[index][1]
                            if pl:
                                # if cell not in correction_confidence or confidence > correction_confidence[cell]:
                                #     correction_confidence[cell] = confidence
                                correction_dictionary[cell] = pc
                s = len(labeled_tuples)
                er = d.evaluate_data_cleaning(correction_dictionary)[-3:]
                aggregate_results[s] = numpy.append(aggregate_results[s], [er], axis=0)
                print(">>>", len(labeled_tuples), er)
        results_string = "\\addplot[error bars/.cd,y dir=both,y explicit] coordinates{(0,0.0)"
        for s in sampling_range:
            mean = numpy.mean(aggregate_results[s], axis=0)
            std = numpy.std(aggregate_results[s], axis=0)
            print("Raha 2 on", d.name)
            print("Labeled Tuples Count = {}".format(s))
            print("Precision = {:.2f} +- {:.2f}".format(mean[0], std[0]))
            print("Recall = {:.2f} +- {:.2f}".format(mean[1], std[1]))
            print("F1 = {:.2f} +- {:.2f}".format(mean[2], std[2]))
            print("--------------------")
            # results_string += "({},{:.2f})+-(0,{:.2f})".format(s, mean[2], std[2])
            results_string += "({},{:.2f})".format(s, mean[2])
        results_string += "}; \\addlegendentry{Raha 2}"
        print(results_string)
########################################


########################################
if __name__ == "__main__":
    # --------------------
    application = Raha2()
    # --------------------
    dataset_name = "flights"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.join(application.DATASETS_FOLDER, dataset_name, "dirty.csv"),
        "clean_path": os.path.join(application.DATASETS_FOLDER, dataset_name, "clean.csv")
    }
    d = dataset.Dataset(dataset_dictionary)
    # --------------------
    # application.revision_extractor()
    # application.value_based_models_pretrainer()
    # detected_data_errors = dict(d.actual_errors_dictionary)
    # detected_data_errors = pickle.load(open("results/error-detection/{}.dictionary".format(d.name), "rb"))
    # application.error_corrector(d, detected_data_errors)
    # --------------------
########################################

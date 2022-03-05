import spacy, benepar
from spacy.matcher import Matcher
from itertools import dropwhile
from queue import Queue
from config import config 


class Path(list):

    def __init__(self):
        super(list, self).__init__()
        self.hashes = set()

    def __contains__(self, item) -> bool:
        return hash(item.text) in self.hashes

    def __delitem__(self, key) -> None:
        self.hashes.remove(self[key])
        super().__delitem__(key)

    def append(self, item) -> None:
        super().append(item)
        self.hashes.add(hash(item.text))

    def extend(self, iterable) -> None:
        super().extend(iterable)
        for item in iterable:
            self.hashes.add(hash(item.text))

    def clear(self) -> None:
        super().clear()
        self.hashes = set()


class Determiner(object):

    def __init__(self, pronoun, vocab):
        gender       = pronoun.morph.get("Gender")[0]
        number       = pronoun.morph.get("Number")[0]
        pattern      = [{
                            "MORPH": {
                                "IS_SUPERSET": [
                                    "Number={}".format(number), 
                                    #"Gender={}".format(gender),
                                ],
                            },
                            "POS": {
                                "IS_SUBSET": ["PROPN", "NOUN"],
                            },
                        }]
        self.matcher = Matcher(vocab)
        self.matcher.add("NP Finder", [pattern])

    def agrees(self, phrase) -> bool:
        """
        Description
        -----------
            Extremely basic function to determine if pronoun gender matches gender in noun phrase.

        Returns  
        ------- 
            agreement : boolean value representing that the pronoun agrees or disagrees with the noun phrase.
        """
        matches = self.matcher(phrase)
        return len(matches) > 0


def hobbs(pronoun, doc) -> str:
    """
    Description
    -----------
        Algorithm to determine the noun phrase the pronoun is representing.

    Arguments
    ---------- 
        pronoun : SpaCy token or string from document.
        doc     : SpaCy document or string; sentences up to and including the sentence containing the pronoun in question.

    Returns  
    ------- 
        antecedent : the noun phrase that the prounoun refers to.
    """
    nlp = spacy.load(config["language model"], disable=config["disable"])
    # Berkley neural parser for constituency parsing.
    nlp.add_pipe("benepar", config=config["benepar config"])
    if isinstance(doc, str):
        doc = nlp(doc)
    elif isinstance(doc, spacy.tokens.doc.Doc):
        # Need to re-parse to get constituencies.
        doc = nlp(doc.text)

    if isinstance(pronoun, str):
        matcher = Matcher(nlp.vocab)
        pattern = [{"ORTH": pronoun, "POS": "PRON"}]
        matcher.add("GetPronoun", [pattern])
        matches = matcher(doc)
        try:
            _, i, _ = matches[0]
            pronoun = doc[i]
        except IndexError:
            raise ValueError(("There is no '{}' pronoun found in the text document given. " 
                                "Please check that the spelling and capitalization are an exact match.").format(pronoun))

    sentences = list(doc.sents)
    node      = pronoun
    path      = Path()
    dt        = Determiner(pronoun, nlp.vocab)
    path.append(node)
    # Step 1.
    # Begin at the noun phrase (NP) node immediately dominating the pronoun.
    label = node._.labels
    while label[0] != 'NP':
        node  = node._.parent
        label = node._.labels
        path.append(node)              # Store path for later.

    # Step 2. 
    # Go up the tree to the first node encountered that is a NP or sentence (S) and set as node.
    node  = node._.parent
    label = node._.labels
    path.append(node)
    while label and label[0] not in {'NP','S'}:
        node  = node._.parent
        label = node._.labels
        path.append(node)             

    # Step 3. 
    # Traverse all branches below node to the left of path in breadth first, left-to-right fashion.
    # Propose as antecedent any encountered NP node that has a NP or S node b/t it and node.
    visited = Queue()
    for visited_node in node._.children:
        visited.put(visited_node)
        # Make sure not to put nodes right of path in queue to be explored.
        if visited_node in path:
            break

    while not visited.empty():
        explored_node  = visited.get()
        explored_label = explored_node._.labels
        if explored_node not in path and explored_label and explored_label[0] == 'NP' and dt.agrees(explored_node):
            # Check for NP or S b/t it and node.
            tmp = explored_node._.parent
            while tmp != node:
                if tmp._.labels[0] in {'NP', 'S'}:
                    return explored_node.text
                tmp = tmp._.parent

        for visited_node in explored_node._.children:
            visited.put(visited_node)
            if visited_node in path:
                break

    while True:
        # Step 4.
        # If node is highest S node in sentence, traverse the surface parse trees of previous sentences in order of recency
        # in breadth first, left-to-right manner. Propose any NP encountered.
        label = node._.labels
        if label and label[0] == 'S' and not node._.parent:
            # Iterate in reverse order, excluding the last sentence.
            for sent in sentences[-2::-1]:
                for visited_node in sent._.children:
                    visited.put(visited_node)

                while not visited.empty():
                    explored_node  = visited.get()
                    explored_label = explored_node._.labels
                    if explored_label and explored_label[0] == 'NP' and dt.agrees(explored_node):
                        return explored_node.text

                    for visited_node in explored_node._.children:
                        visited.put(visited_node)

        # Step 5.
        # From node, go up tree to first NP or S and set as node. 
        node  = node._.parent if node._.parent else node
        label = node._.labels
        path.clear()
        path.append(node)
        while label and label[0] not in {'NP','S'}:
            node  = node._.parent
            label = node._.labels
            path.append(node)

        # Step 6.
        # If node is an NP and if path did not pass through a Nominal that is immediately dominated by node, propose node.
        if label[0] == 'NP':
            dominated_nominals = map(lambda x: x._.labels[0] == 'NN' and x in path, node._.children)
            if not any(dominated_nominals) and dt.agrees(node):
                return node.text

        # Step 7.
        # Traverse breadth-first, left-to-right, all branches below node to the left of path. Propose any NP as the antecedent. 
        for visited_node in node._.children:
            visited.put(visited_node)
            # Make sure not to put nodes right of path in queue to be explored.
            if visited_node in path:
                break

        while not visited.empty():
            explored_node  = visited.get()
            explored_label = explored_node._.labels
            if explored_node not in path and explored_label and explored_label[0] == 'NP' and dt.agrees(explored_node):
                return explored_node.text

            for visited_node in explored_node._.children:
                visited.put(visited_node)
                if visited_node in path:
                    break

        # Step 8.
        # If node is an S, traverse breadth-first, left-to-right, all branches of node to the right of path. 
        # Don't go below any NP or S encountered. Propose any seen NP.
        if node._.labels[0] == 'S':
            # Node is last thing put in path, so we can start there.
            for path_node in path[::-1]:
                # Drop all children left of path.
                for visited_node in dropwhile(lambda x: x not in path, path_node._.children):
                    visited.put(visited_node)

                while not visited.empty():
                    explored_node  = visited.get()
                    explored_label = explored_node._.labels
                    if explored_node not in path and explored_label and explored_label[0] == 'NP' and dt.agrees(explored_node):
                        return explored_node.text

                    for visited_node in dropwhile(lambda x: x not in path, explored_node._.children):
                        visited.put(visited_node)

        # Step 9.
        # Go to Step 4.


if __name__ == '__main__':
    # Some example sentences. Hobbs used s3 as an example where the algorithm was incorrect.
    s1 = "Nick rushed out the door. He did not want to miss his morning flight."
    s2 = "Nick's grandfather, an older gentleman born in the forties, jumped the fence while the girl was watching. As he went over, it scraped his leg."
    s3 = "The castle in Camelot remained the residence of the king until 536 when he moved it to London."
    s4 = "Natalie rushed out the door. She did not want to miss her morning flight."

    doc        = s1
    pronoun    = "He"
    antecedent = hobbs(pronoun, doc)
    print(pronoun, antecedent)
import sys

def postprocess(raw_result: str):
    # Remove unnecessary tags
    result = raw_result.replace("<pad>", "")
    result = result.replace("</s>", "")
    result = result.strip()
    return delinearize_2(result)

def delinearize(linearized_amr: str) -> str:
    """
    Unused. Use delinearize_2 instead.
    """

    concepts = []
    tokens = extract_tokens(linearized_amr)
    for token in tokens:
        if is_concept(token) and token not in concepts:
            concepts.append(token)

    concept_next_index = {}
    concept_to_node_map = {}
    for concept in concepts:
        candidate_node_name = get_candidate_node_name(token)
        if candidate_node_name not in concept_next_index.keys():
            concept_to_node_map[concept] = candidate_node_name
            concept_next_index[candidate_node_name] = 2
        else:
            concept_to_node_map[concept] = candidate_node_name + str(concept_next_index[candidate_node_name])
            concept_next_index[candidate_node_name] += 1

    new_tokens = []
    used_concepts = []
    skip_until_closed_bracket_count = 0
    level = 0
    for token in tokens:
        if token == "(":
            level += 1
        elif token == ")":
            level -= 1

        if skip_until_closed_bracket_count > 0:
            if token == ")":
                skip_until_closed_bracket_count -= 1
            elif token == "(":
                skip_until_closed_bracket_count += 1

            continue

        if token in concept_to_node_map.keys():
            if token not in used_concepts:
                used_concepts.append(token)
                new_tokens.append(concept_to_node_map[token])
                new_tokens.append("/")  
                new_tokens.append(token)
            else:
                new_tokens.pop()
                new_tokens.append(concept_to_node_map[token])
                skip_until_closed_bracket_count = 1
        else:
            new_tokens.append(token)

        if level == 0:
            break

    assert skip_until_closed_bracket_count == 0
    
    return " ".join(new_tokens)

def extract_tokens(linearized_amr: str) -> list[str]:
    result_tokens = linearized_amr.split(" ")
    new_result_tokens: list[str] = []
    for token in result_tokens:
        new_token = ""
        for c in token:
            if c in ["(", ")"]:
                if new_token != "":
                    new_result_tokens.append(new_token)
                    new_token = ""

                new_result_tokens.append(c)
            else:
                new_token += c

        if new_token != "":
            new_result_tokens.append(new_token)

    result_tokens = new_result_tokens
    new_result_tokens = []

    status: str = "normal"
    current_concept_words = []
    for token in result_tokens:
        if status == "normal":
            if is_concept(token):
                current_concept_words = [token]
                status = "building_concept"
            else:
                new_result_tokens.append(token)
        elif status == "building_concept":
            if is_concept(token):
                current_concept_words.append(token)
            else:
                new_result_tokens.append("_".join(current_concept_words))
                current_concept_words = []
                status = "normal"
                new_result_tokens.append(token)
        else:
            raise ValueError
        
    if status == "building_concept":
        new_result_tokens.append("_".join(current_concept_words))

    result_tokens = new_result_tokens
    return result_tokens

def is_concept(token):
    return token not in ["(", ")"] and (not is_relation(token))

def is_relation(token):
    return token.startswith(":")

def delinearize_2(linearized_amr: str, keep_duplicated_relation: bool = False) -> str:
    tokens = extract_tokens(linearized_amr)
    new_tokens = []
    level = 0

    concept_next_index = {}
    concept_stack = []
    used_relations: list[tuple[str, str]] = []

    status = "normal"
    for token in tokens:
        if token == "(":
            level += 1
        elif token == ")":
            level -= 1

        if status == "normal":
            if is_concept(token):
                candidate_node_name = get_candidate_node_name(token)
                if candidate_node_name not in concept_next_index.keys():
                    node_name = candidate_node_name
                    concept_next_index[candidate_node_name] = 2
                else:
                    node_name = candidate_node_name + str(concept_next_index[candidate_node_name])
                    concept_next_index[candidate_node_name] += 1

                concept_stack.append(node_name)
                new_tokens.append(node_name)
                new_tokens.append("/")
                new_tokens.append(token)

            elif is_relation(token):
                if keep_duplicated_relation:
                    new_tokens.append(token)
                elif len(concept_stack) > 0:
                    current_relation = (concept_stack[-1], token)
                    if current_relation not in used_relations:
                        new_tokens.append(token)
                        used_relations.append(current_relation)
                    else:
                        status = f"find_closed_parentheses_level_{level}"
                else:
                    raise ValueError("What, no concept?")

            else:
                if token == ")":
                    concept_stack.pop()

                new_tokens.append(token)

        elif status.startswith("find_closed_parentheses_level"):
            if token == ")" and f"find_closed_parentheses_level_{level}" == status:
                status = "normal"

        if level == 0:
            break

    while level > 0 and status == "normal":
        new_tokens.append(")")
        level -= 1
    
    return " ".join(new_tokens)

def get_candidate_node_name(token):
    if token[0] >= "a" and token[0] <= "z": 
        return token[0]
    else:
        return "sym"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raw_result = "<pad> ( dan :op1 ( jarak :arg0 ( jarak :mod ( sekolah ) ) ) :arg1 ( rumah :mod ( aku ) ) ) )</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"
        result = postprocess(raw_result)
        print(result)
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        with open(input_path, mode="r") as fi:
            lines = fi.readlines()
        
        output_lines = []
        for index, line in enumerate(lines):
            print(f"Line {index + 1}")

            line = line.strip()
            #print("Input:", line)
            result = postprocess(line)
            output_lines.append(result)
            #print("Output:", result)
            #print()

        with open(output_path, mode="w") as fo:
            for line in output_lines:
                print(line, file=fo)
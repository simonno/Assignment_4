def append_prefix_key(features, prefix):
    return {prefix + k: v for k, v in features.items()}


def get_right_and_left_tokens(entity):
    sent = entity.doc
    for i in range(len(sent)):
        if entity.text.startswith(sent[i].text):
            right_index = len(entity.text.split())
            if i + right_index < len(sent):
                return list(reversed(sent[:i])), sent[i + right_index:]
            else:
                return list(reversed(sent[:i])), []


def token_feature(token):
    if token is None:
        return {}
    return {
        'type': token.ent_type_,
        'root_dep': token.dep_,
        'pos': token.pos_,
        'tag': token.tag_,
        'is_lower': token.is_lower,
        'iob': token.ent_iob_,
    }


def neighbors_features(tokens_list, prefix='', neighbors=1):
    features = dict()
    if neighbors > len(tokens_list):
        neighbors = len(tokens_list)

    for n in range(neighbors):
        features.update(append_prefix_key(token_feature(tokens_list[n]), '{}{}_'.format(prefix, n)))

    return features


def entity_features(entity):
    left_tokens, right_tokens = get_right_and_left_tokens(entity)
    features = token_feature(entity.root)
    features['title'] = entity.root.is_title
    features['quote'] = entity.root.is_quote
    features['digit'] = entity.root.is_digit
    # features['cluster'] = entity.root.cluster
    # features['oov'] = entity.root.is_oov
    features.update(neighbors_features(right_tokens, 'r'))
    features.update(neighbors_features(left_tokens, 'l'))
    return features


#     return {}
# 'is_lower': entity.root.is_lower,
# 'n_lefts': entity.root.n_lefts,
# 'le_dep': token.left_edge.dep_,
# 'le_type': token.left_edge.ent_type_,
# 'le_pos': token.left_edge.pos_,
# 'n_rights': token.n_rights,
# 're_dep': token.right_edge.dep_,
# 're_type': token.right_edge.ent_type_,
# 're_pos': token.right_edge.pos_,
# 'prob': token.prob,
# '2_lower': token.lower_,
# 'shape': token.shape_


def get_features(first_ent, second_ent):
    features = append_prefix_key(entity_features(first_ent), '1_')
    features.update(append_prefix_key(entity_features(second_ent), '2_'))
    return features

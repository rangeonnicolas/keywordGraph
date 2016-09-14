import json
import pandas as pd
import re
import operator
import networkx as nx
import community

class KeywordGraph:

    keywords = None

    bipartite = None

    nodes = None

    max_edge_weight_by_word = None

    word_cnt = None
    edges_unfiltered = None
    edges_filtered = None
    max_edge_weight_by_doc = None

    nxGraph = None
    clusters = None

    def __init__(self, keywords):
        self.keywords = keywords

    def get_keywords(self):
        return(self.keywords)

    def get_bipartite(self, keep_in_memory= False):
        if self.bipartite:
            return self.bipartite
        else:
            data = self._compute_bipartite(self.keywords)
            if keep_in_memory:
                self.bipartite = data
            return data

    def get_nodes(self, label= True, keep_in_memory= False):
        if self.nodes:
            return self.nodes
        else:
            data = self._compute_nodes(self.keywords, label= label)
            if keep_in_memory:
                self.nodes = data
            return data

    def get_edges(self, min_nb_of_texts_for_each_word, edge_list_must_cover_all_texts= False, edge_list_length= None
                  , keep_in_memory= False):
        if self.edges_filtered:
            return self.edges_filtered
        else:

            if not self.word_cnt:
                _, word_cnt= self._compute_nodes(self.keywords)

            edges_unfiltered, max_edge_weight_by_doc = self._compute_edges(self.keywords, word_cnt,
                                                                           min_nb_of_texts_for_each_word)
            edges_filtered = self._filter_edges(edges_unfiltered, max_edge_weight_by_doc,
                                 edge_list_must_cover_all_texts= edge_list_must_cover_all_texts,
                                 edge_list_length = edge_list_length)

            if keep_in_memory:
                self.edges_unfiltered = edges_unfiltered
                self.word_cnt = word_cnt
                self.edges_filtered = edges_filtered
                self.max_edge_weight_by_doc = max_edge_weight_by_doc

            return edges_filtered

    def get_max_edge_weight_by_word(self, keep_in_memory= False, all_the_words= False):
        if self.max_edge_weight_by_word:
            return self.max_edge_weight_by_word
        else:
            try:
                edges = self.edges_unfiltered
            except:
                raise BaseException("""Edges have not been computed yet.
                Please call get_edges(... , keep_in_memory= True) before calling get_max_edge_weight_by_word""")

            data = self._compute_max_edge_weight_by_word(edges)

            if all_the_words:
                if self.bipartite is None:
                    bipartite = self.get_bipartite(keep_in_memory= keep_in_memory)
                else:
                    bipartite = self.bipartite
                for word in bipartite["mc"].values:               # todo: mettre un unique()
                    if word not in data.keys():
                        data[word]= None

            if keep_in_memory:
                self.max_edge_weight_by_word = data
            return data

    def compute_clusters(self, min_nb_of_texts_for_each_word = None, edge_list_must_cover_all_texts= None,
                         edge_list_length= None, for_gephi = None, keep_in_memory= False):

        if not self.clusters is None:

            if min_nb_of_texts_for_each_word or edge_list_must_cover_all_texts or edge_list_length or for_gephi:
                raise BaseException("""Clusters have already been computed. These parameters are useless :
                                    min_nb_of_texts_for_each_word, edge_list_must_cover_all_texts, edge_list_length,
                                    for_gephi. Please remove them, or specify parameter 'keep_in_memory = False'
                                    when first calling method 'compute_clusters'""")
            return self.clusters

        else :

            # Valeurs par default
            if not min_nb_of_texts_for_each_word :
                min_nb_of_texts_for_each_word = 1
            if not edge_list_must_cover_all_texts:
                edge_list_must_cover_all_texts = False
            if not for_gephi:
                for_gephi = False

            # Calcul des liens du graphe
            edges = self.get_edges(min_nb_of_texts_for_each_word,
                                   edge_list_must_cover_all_texts= edge_list_must_cover_all_texts,
                                   edge_list_length= edge_list_length, keep_in_memory= keep_in_memory)

            # Creation d'un graphe NetworkX
            G = nx.Graph()
            G.add_weighted_edges_from(edges)
            if G.is_directed() :
                G = G.to_undirected()

            # Calcul des clusters
            dendo = community.generate_dendogram(G)

            # Calcul de la betweeness centrality
            #betCen = nx.betweenness_centrality(G,weight="weight")
            betCen = nx.betweenness_centrality(G) # c'est mieux comme ca... un peu arbitrairement d'ailleurs...

            # Mise en forme
            res = pd.DataFrame(pd.Series(community.best_partition(G)).order())
            res.columns = ["cluster"]
            res["label"] = res.index
            if for_gephi :
                res["id"] = res.index
            res["betcen"] = 0
            for wor in G.nodes():
                res.ix[wor,"betcen"] = betCen[wor]
            res = res.sort(["cluster","betcen"],ascending=[True,False])

            # transformation des noms des clusters en strings
            res.cluster = ["cl_" + str(s) for s in res.cluster]

            # Enregistrement
            if keep_in_memory:
                self.nxGraph = G                        # todo: checker l'utilité
                self.clusters = res

            return res

    def get_docs_with_keyword_clusters(self, min_nb_of_texts_for_each_word = None, edge_list_must_cover_all_texts = None,
                                       edge_list_length = None, names= True):

        # todo: mode keep_in_memory

        clusters = self.compute_clusters(min_nb_of_texts_for_each_word, edge_list_must_cover_all_texts,
                         edge_list_length)

        # Table d'association document<->mot-clé
        bipartite = self.get_bipartite()

        # donne a chaque cluster de mot le nom des 10 premiers mots du cluster
        clusters_with_names, cluster_names = self._give_a_name_to_clusters(clusters)

        # Table d'association document<->cluster de mot-clés
        if names :
            result = pd.merge(clusters_with_names, bipartite, left_on= "label", right_on= "word")[["id_doc","name"]]
            result.columns = ["id_sheet","cluster_name"]
        else:
            result = pd.merge(clusters, bipartite, left_on= "label", right_on= "word")[["id_doc","cluster"]]
            result.columns = ["id_sheet","cluster_id"]

        return result, cluster_names

    def _give_a_name_to_clusters(self, clusters) :

        clus_names = pd.DataFrame(columns=["cluster","name"])
        clus_names["cluster"] = clusters.cluster.unique()
        clus_names.index = clus_names["cluster"]
        for cl in clus_names["cluster"] :
            st = ""
            subset = clusters.ix[clusters.cluster == cl]
            subset.index = range(len(subset.index))
            for i in range(10) :
                try:
                    st += "/" + subset.label[i]
                except KeyError:
                    pass
            clus_names.ix[cl,"name"] = st
        clusters_with_names = pd.merge(clus_names,clusters,on=["cluster"])

        return clusters_with_names, clus_names

    def _filter_edges(self, edges, max_edge_weight_by_doc, edge_list_must_cover_all_texts= False, edge_list_length= None):
        # edge_list_must_cover_all_texts is ignored if edge_list_length is None
        if edge_list_length is None and edge_list_must_cover_all_texts == False:
            return self._convert_edges_to_list_of_tuples(edges)

        if edge_list_length is None and edge_list_must_cover_all_texts == True:
            edge_list_length = 1

        if edge_list_must_cover_all_texts == False:
            return self._get_n_firsts(edges, edge_list_length)
        else:
            min_weight_to_get_all_texts = self._min_weight_of_maxEdgeWeightByDoc(max_edge_weight_by_doc)
            min_number_of_edges_to_keep = self._how_many_edges_to_reach_a_given_weight(edges, min_weight_to_get_all_texts)
            nb_to_keep = max(min_number_of_edges_to_keep, edge_list_length)
            return self._get_n_firsts(edges, nb_to_keep)

    def _convert_edges_to_list_of_tuples(self, edges, ordered_keys= None):
        if not ordered_keys:
            ordered_keys = edges.keys()

        # These lines just change the form of 'edges' from :
        # {('sommet', 'forme'): 15.424, ('religieuse', 'transept'): 59.895, ...}
        # to
        # [('sommet', 'forme', 15.424), ('religieuse', 'transept', 59.895), ... ]
        edges_list_of_tuples = []
        for key in ordered_keys:
            edges_list_of_tuples = edges_list_of_tuples + [(key[0], key[1], edges[key])]
        return edges_list_of_tuples


    def _how_many_edges_to_reach_a_given_weight(self, edges, weight):
        nb = 0
        for edge in edges:
            if edges[edge] >= weight :
                nb = nb+1
        return nb

    def _compute_bipartite(self, keywords):

        bipart = []
        for id_doc in keywords.keys():
            for word_ind in keywords[id_doc].keys():
                word   = keywords[id_doc][word_ind]["word"]
                weight = keywords[id_doc][word_ind]["weight"]
                bipart = bipart + [(id_doc, word, weight)]

        df = pd.DataFrame(bipart)
        df.columns = ["id_doc","word","weight"]

        return df

    def _compute_nodes(self, keywords, label= True):

        word_cnt = dict()
        for id_doc in keywords.keys():
            for word_ind in keywords[id_doc].keys():
                word = keywords[id_doc][word_ind]["word"]

                if word == "demandent":
                    print("--------------", word)

                if word in word_cnt.keys():
                    word_cnt[word] = word_cnt[word] +1
                else:
                    word_cnt[word] = 1

                if word == "demandent":
                    print("--------res--",word_cnt[word])

        ser = pd.Series(word_cnt)
        df = pd.DataFrame(ser, index= ser.index, columns= ["cnt"])
        df["id"] = df.index
        if label:                                   # Gephi-like format
            df["label"] = df["id"]                  # Gephi-like format

        return df, word_cnt

    def _compute_edges(self, keywords, word_cnt, min_freq, nb_decimals= 3):

        max_edge_weight_by_doc = dict()                                                                                     # stocke pour chaque document le edge de poids maximal
                                                                                                                            # forme : {'d_559925': {'couple': ('basalte', 'création'), 'weight': 83.759}, 'd_560124': {'couple': ('ulm', 'passagers'), 'weight': 90.557}, 'd_557938': {'couple': ('maxi', 'aéroclub')}
        couple_weight = dict()                                                                                              # forme : {('offre', 'croisières'): 86.076, ('champ', 'mois'): 39.213, ('escalade', 'encadrement'): 35.65, ('construit', '1500'): 21.431}
        for id_doc in keywords.keys():
            keywords_id_doc_keys = keywords[id_doc].keys()
            for word_ind1 in keywords_id_doc_keys:
                word1 = keywords[id_doc][word_ind1]["word"]
                if type(word1) == int:
                    raise BaseException("word1 should not be an int, il should be a string")
                if word_cnt[word1] >= min_freq :
                    for word_ind2 in keywords_id_doc_keys:
                        word2 = keywords[id_doc][word_ind2]["word"]
                        if type(word2) == int:
                            raise BaseException("word2 should not be an int, il should be a string")
                        if word_cnt[word2] >= min_freq :
                            if word_ind1> word_ind2:                                                                        # pour ne pas avoir de doublons du type (w1,w2) et (w2,w1)
                                key = (word1, word2)
                                score = round(
                                    keywords[id_doc][word_ind1]["weight"] * keywords[id_doc][word_ind2]["weight"]           # le score d'un couple de mots est égal au prduit de leurs poids respectifs (oui ok c'est un peu arbitraire mais bon...)
                                    , nb_decimals)
                                if key in couple_weight.keys() :                                                            # si le couple a déjà été trouvé dans un autre id_doc
                                    new_score = couple_weight[key] + score
                                    couple_weight[key] = new_score
                                    self._add_couple_to_maxEdgeWeightByDoc_if_needed(key, new_score, id_doc,
                                                                                max_edge_weight_by_doc, updated_score= True)
                                else:
                                    couple_weight[key] = score
                                    self._add_couple_to_maxEdgeWeightByDoc_if_needed(key, score, id_doc,
                                                                                max_edge_weight_by_doc)
        return couple_weight, max_edge_weight_by_doc

    def _compute_max_edge_weight_by_word(self, edges):
        max_weight = dict()
        for couple in edges.keys():
            word1 = couple[0]
            word2 = couple[1]
            if word1 in max_weight.keys():
                max_weight[word1] = max(max_weight[word1], edges[couple])
            else:
                max_weight[word1] = edges[couple]
            if word2 in max_weight.keys():
                max_weight[word2] = max(max_weight[word2], edges[couple])
            else:
                max_weight[word2] = edges[couple]
        return(max_weight)

    def _min_weight_of_maxEdgeWeightByDoc(self, max_edge_weight_by_doc):
            keys = list(max_edge_weight_by_doc.keys())
            min_weight_to_get_all_texts = max_edge_weight_by_doc[keys[0]]["weight"]
            for id_doc in keys[1:]:
                if max_edge_weight_by_doc[id_doc]["weight"] < min_weight_to_get_all_texts:
                    min_weight_to_get_all_texts = max_edge_weight_by_doc[id_doc]["weight"]
            return min_weight_to_get_all_texts


    def _get_n_firsts(self, edges, edge_list_length, order = False):

        if len(edges) <= edge_list_length:
            if order:
                ordered_keys = sorted(edges,key=edges.get,reverse=True)
                return self._convert_edges_to_list_of_tuples(edges, ordered_keys= ordered_keys) #todo: non testé
            else:
                return self._convert_edges_to_list_of_tuples(edges) #todo: non testé
        else:
            size = len(edges)
            edges_keys = list(edges.keys())

            edges_temp = {}
            for key in edges_keys[0:edge_list_length] :
                edges_temp[key] = edges[key]
            sorted_keys = sorted(edges_temp, key= edges_temp.get, reverse = True)

            ordered = self._convert_edges_to_list_of_tuples(edges_temp, ordered_keys= sorted_keys)

            min_of_ordered = ordered[-1][2]

            for couple in edges_keys[(edge_list_length):]:
                if edges[couple] > min_of_ordered:
                    # init
                    ind = 0
                    # determination of where_to_put
                    while edges[couple] < ordered[ind][2]:
                        ind = ind +1
                    where_to_put = ind
                    # tuple to add in 'ordered'
                    to_add = (couple[0], couple[1], edges[couple])
                    # insertion
                    for i in range(where_to_put, edge_list_length):
                        aux1 = ordered[i]
                        ordered[i] = to_add
                        to_add = aux1
                    min_of_ordered = ordered[-1][2]
            return ordered




    def _add_couple_to_maxEdgeWeightByDoc_if_needed(self, couple, score, id_doc, max_edge_weight_by_doc, updated_score= False):
        if id_doc not in max_edge_weight_by_doc.keys():
            max_edge_weight_by_doc[id_doc] = dict()
            max_edge_weight_by_doc[id_doc]["weight"] = score
            max_edge_weight_by_doc[id_doc]["couple"] = couple

        elif score > max_edge_weight_by_doc[id_doc]["weight"] :
            max_edge_weight_by_doc[id_doc]["weight"] = score
            max_edge_weight_by_doc[id_doc]["couple"] = couple

        # si "couple" a déjà été analysé lors d'une précédente itération, et a été retenu comme le couple de poids
        # maximal pour un texte, alors il faut actualiser ce poids car celui-ci vient de changer lors de l'appel
        # des lignes
        # new_score = couple_weight[key] + score
        # couple_weight[key] = new_score
        # de la fonction appelante
        if updated_score :
            for id in max_edge_weight_by_doc.keys():
                if id != id_doc and max_edge_weight_by_doc[id]["couple"] == couple :
                    max_edge_weight_by_doc[id]["weight"] = score

    def get_gephi_edges_table(self, min_nb_of_texts_for_each_word, edge_list_must_cover_all_texts= False,
                              edge_list_length= None, keep_in_memory= False):

            edges = self.get_edges(min_nb_of_texts_for_each_word, edge_list_must_cover_all_texts, edge_list_length,
                                   keep_in_memory= keep_in_memory)

            df = pd.DataFrame(edges, columns= ["Source","Target","Weight"])
            print(df.ix[0:10,:])

            return df




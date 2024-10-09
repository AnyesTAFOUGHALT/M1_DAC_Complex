import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from queue import LifoQueue
import time
import math
import random 
###################################################################################
###                              Partie 2 : Graphe                              ###    
###################################################################################


# 2.1.1           
def delete_node (G , n) :
    """
    Cette fonction supprime le noeud n du graphe G
        G : un graphe
        n : un noeud à supprimer du graphe
        return : le graph G sans le noeud n
    """
    G2 = G.copy()
    G2.remove_node(n)
    return G2

# 2.1.2
def delete_nodes(G , nodes):
    """
    Cette fonction supprime les noeuds nodes du graphe G
        G : un graphe
        nodes : la liste des noeuds à supprimer du graphe
        return : le graph G sans les noeuds nodes
    """
    G2 = G.copy()
    for n in nodes :
        G2 = delete_node( G2 , n)
    return G2

# 2.1.3
def degree_nodes (G):
    """
        G : un graphe
        return : une liste qui contient les degré de chaque noeud
    """
    return [G.degree[n] for n in G.nodes]

def max_degree(G) : 
    """
        G : un graphe
        return : le noeud qui a le degré maximum
    """
    degrees = degree_nodes (G)
    return list(G.nodes)[degrees.index(max(degrees))]

# 2.2.1
def generate_alea_graphe(n , p):
    """
        n : nombre de sommets
        p = une probabilité
        return : un graphe génèré aléatoirement avec n nœuds, où la probabilité d’avoir une arête entre chaque paire de nœuds est p
    """
    return nx.fast_gnp_random_graph(n, p)

###################################################################################
###                        Partie 3 : Méthodes approchées                       ###    
###################################################################################

#3.2
def algo_couplage(G):
    """
        G : un graphe
        return : la couverture retournée par l'algorithme de couplage
    """
    nodes = []  
    for e in G.edges:
        if e[0] not in nodes and e[1] not in nodes:
            nodes.append(e[0])
            nodes.append(e[1])
    return nodes

def algo_glouton(G) :
    """
        G : un graphe
        return : la couverture retournée par l'algorithme de glouton
    """
    E = G.copy()
    C = []
    while(len(E.edges) != 0) :
        n = max_degree(E)
        C.append(n)
        E = delete_node (E , n)
    return C

def compare_couplage_and_glouton(nMin , nMax , nb_iter = 10, nb_mean = 10 , show = True ):
    """
        nMin : le nombre de sommet minimum
        nMax : le nombre de sommet maximum
        nb_iter : le parcours de noeuds se fait avec un pas de nMax/nb_iter
        nb_mean : pour chaque n et p on génère nb_mean graphes pour calculer la moyenne
        show : True si on veut afficher le graphe, False sinon
        
        Cette fonction crée des graphes afin de comparer le temps d'execution et la taille de couveture des algorithmes de glouton et couplage en fonction de p et n
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    length_couverture_couplege = []
    execution_time_couplage = []
    length_couverture_glouton = []
    execution_time_glouton = []

    nb_nodes = np.array(range(nMin , nMax+1 ,nMax// nb_iter))

    # On parcourt les p suivant : [0.25 , 0.5 , 0.75  , 1]
    Probabilities =  [0.25 , 0.5 , 0.75  , 1]
    for p in range(len(Probabilities)):
        length_couverture_couplege_n = []
        execution_time_couplage_n = []
        length_couverture_glouton_n = []
        execution_time_glouton_n = []
        
        for n in nb_nodes :
            length_couverture_couplage_i = 0
            execution_time_couplage_i = 0
            length_couverture_glouton_i = 0
            execution_time_glouton_i = 0
            for i in range(nb_mean):
                #On génére un graphe de taille n et de probabilité p
                graph = generate_alea_graphe(n , Probabilities[p])
                
                #On calcul le temps d'execution de l'algorithme de couplage ainsi que la taille de sa couverture
                time_start = time.time()
                couverture_couplage = algo_couplage(graph)
                execution_time_couplage_i += time.time() - time_start
                length_couverture_couplage_i += len(couverture_couplage)

                #On calcul le temps d'execution de l'algorithme de glouton ainsi que la taille de sa couverture
                time_start = time.time()
                couverture_glouton = algo_glouton(graph)
                execution_time_glouton_i += time.time() - time_start
                length_couverture_glouton_i += len(couverture_glouton)
            
            #On fait les calcul pour une approximation de nb_mean executions
            length_couverture_couplege_n.append(length_couverture_couplage_i/nb_mean)
            execution_time_couplage_n.append(execution_time_couplage_i/nb_mean)
            length_couverture_glouton_n.append(length_couverture_glouton_i/nb_mean)
            execution_time_glouton_n.append(execution_time_glouton_i/nb_mean)

        length_couverture_couplege.append(length_couverture_couplege_n)
        execution_time_couplage.append(execution_time_couplage_n)
        length_couverture_glouton.append(length_couverture_glouton_n)
        execution_time_glouton.append(execution_time_glouton_n)

        if p < 2 :
            axs[0, p%2].plot(nb_nodes, execution_time_couplage_n, 'r', label="Algo Couplage")
            axs[0, p%2].plot(nb_nodes, execution_time_glouton_n, 'b', label="Algo Glouton")
            axs[0, p%2].set_xlabel('Nombre de sommets')
            axs[0, p%2].set_ylabel('Temps d\'execution')
            axs[0, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs[0, p%2].legend()

        else :
            axs[1, p%2].plot(nb_nodes, execution_time_couplage_n, 'r', label="Algo Couplage")
            axs[1, p%2].plot(nb_nodes, execution_time_glouton_n, 'b', label="Algo Glouton")
            axs[1, p%2].set_xlabel('Nombre de sommets')
            axs[1, p%2].set_ylabel('Temps d\'execution')
            axs[1, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs[1, p%2].legend()
    fig.suptitle("Comparaison du temps d\'execution des algorithemes de couplage et de glouton" )
    plt.tight_layout()
    plt.savefig("Plots/Temps_de_calcul/Comparaison du temps d\'execution entre l\'algorithme de couplage et de glouton avec n ∈ ["+str(nMin)+","+str(nMax)+"].png")
    
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 9))
    for p in range(len(Probabilities)):
        if p < 2 :
            axs2[0, p%2].plot(nb_nodes, length_couverture_couplege[p], 'r', label="Algo Couplage")
            axs2[0, p%2].plot(nb_nodes, length_couverture_glouton[p], 'b', label="Algo Glouton")
            axs2[0, p%2].set_xlabel('Nombre de sommets')
            axs2[0, p%2].set_ylabel('Taille de la couverture')
            axs2[0, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs2[0, p%2].legend()

        else :
            axs2[1, p%2].plot(nb_nodes, length_couverture_couplege[p], 'r', label="Algo Couplage")
            axs2[1, p%2].plot(nb_nodes, length_couverture_glouton[p], 'b', label="Algo Glouton")
            axs2[1, p%2].set_xlabel('Nombre de sommets')
            axs2[1, p%2].set_ylabel('Taille de la couverture')
            axs2[1, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs2[1, p%2].legend()

    fig2.suptitle("Comparaison de la taille des couverture des algorithemes de couplage et de glouton")
    plt.tight_layout()
    plt.savefig("Plots/Qualite_des_solutions/Comparaison de la taille des couverture entre l\'algorithme de couplage et de glouton avec n ∈ ["+str(nMin)+","+str(nMax)+"].png")
    
    if show :
        plt.show()

###################################################################################
###                   Partie 4 : Séparation et évaluation                     ###    
###################################################################################

#4.1.1
def branchment(Graph) :
    """
    Graph : un graphe
    return : une couverture trouvée par l'algorithme de branchement et le nombre de noeuds visités
    """
    stack = LifoQueue()
    stack.put((Graph.copy() , []))
    Couverture = list(Graph.nodes)
    nb_noeuds_visites = 0
    while(not stack.empty()) :
        nb_noeuds_visites += 1
        # P est un couple (Graphe , couverture)
        G , c = stack.get()
        #Si il n'y a plus d'arrête dans le graphe
        if G.number_of_edges() == 0 :
            #On met à jour la couverture si la taille de la nouvelle couverture trouvée est petite
            if len(c) < len(Couverture):
                Couverture = c
        else :
            edge = list(G.edges)[0]

            fils_Droit = delete_node (G , edge[0])
            c_Droit = c.copy()
            c_Droit.append(edge[0])
            stack.put((fils_Droit , c_Droit))

            fils_Gauche = delete_node (G , edge[1])
            c_Gauche = c.copy()
            c_Gauche.append(edge[1])
            stack.put((fils_Gauche , c_Gauche))
    return Couverture , nb_noeuds_visites

#4.1.2
def analyser_performance_branchement(nMin , nMax , nb_iter = 10, nb_mean = 10 , show = True ):
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    length_couverture = []
    execution_time = []

    nb_nodes = np.array(range(nMin , nMax+1 ,nMax// nb_iter))

    # On parcourt les p suivant : [0.25 , 0.5 , 0.75  , 1]
    Probabilities =  [0.25 , 0.5 , 0.75  , 1]
    for p in range(len(Probabilities)):
        length_couverture_n = []
        execution_time_n = []
        
        for n in nb_nodes :
            length_couverture_i = 0
            execution_time_i = 0
            for i in range(nb_mean):
                print(" p ",p," n ", n,' i ', i)
                #On génére un graphe de taille n et de probabilité p
                graph = generate_alea_graphe(n , Probabilities[p])
                
                #On calcul le temps d'execution de l'algorithme de couplage ainsi que la taille de sa couverture
                time_start = time.time()
                couverture , nb_noeuds_visites = branchment(graph)
                execution_time_i += time.time() - time_start
                length_couverture_i += nb_noeuds_visites
            
            #On fait les calcul pour une approximation de nb_mean executions
            length_couverture_n.append(length_couverture_i/nb_mean)
            execution_time_n.append(execution_time_i/nb_mean)

        length_couverture.append(length_couverture_n)
        execution_time.append(execution_time_n)

        if p < 2 :
            axs[0, p%2].plot(nb_nodes, execution_time_n, 'r', label="Algo Branchement")
            axs[0, p%2].set_xlabel('Nombre de sommets')
            axs[0, p%2].set_ylabel('Temps d\'execution')
            axs[0, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs[0, p%2].legend()

        else :
            axs[1, p%2].plot(nb_nodes, execution_time_n, 'r', label="Algo Branchement")
            axs[1, p%2].set_xlabel('Nombre de sommets')
            axs[1, p%2].set_ylabel('Temps d\'execution')
            axs[1, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs[1, p%2].legend()
    fig.suptitle("Analyse du temps d\'execution de l\'algo de branchement" )
    plt.tight_layout()
    plt.savefig("Plots/Temps_de_calcul/Analyse du temps d\'execution de l\'algo de branchement avec n ∈ ["+str(nMin)+","+str(nMax)+"].png")
    
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 9))
    for p in range(len(Probabilities)):
        if p < 2 :
            axs2[0, p%2].plot(nb_nodes, length_couverture[p], 'r', label="Algo Branchement")
            axs2[0, p%2].set_xlabel('Nombre de sommets')
            axs2[0, p%2].set_ylabel('Nombre de noeuds visités')
            axs2[0, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs2[0, p%2].legend()

        else :
            axs2[1, p%2].plot(nb_nodes, length_couverture[p], 'r', label="Algo Branchement")
            axs2[1, p%2].set_xlabel('Nombre de sommets')
            axs2[1, p%2].set_ylabel('Nombre de noeuds visités')
            axs2[1, p%2].set_title("p={:.2f}".format(Probabilities[p]))
            axs2[1, p%2].legend()

    fig2.suptitle("Analyse de le nombre de noeuds visités dans l\'algo de branchement")
    plt.tight_layout()
    plt.savefig("Plots/Qualite_des_solutions/Analyse du nombre de noeuds visités dans l\'algo de branchement avec n ∈ ["+str(nMin)+","+str(nMax)+"].png")
    
    if show :
        plt.show()

###################################################################################
###                       Partie 4.2 : Ajout de bornes                          ###    
###################################################################################

#4.2.2

def bornes(G , M):
    """
    Graph : un graphe
    M : le couplage du graphe G
    return : les bornes b1 , b2 et b3
    """
    n=G.number_of_nodes()
    m=G.number_of_edges()
    delta = max( degree_nodes (G))

    b1 = m // delta
    b2 = len(M)/2  
    b3 = (2*n-1-math.sqrt((2*n-1)**2-8*m))//2

    return b1 , b2 , b3

def branchement_bornes(Graph) :
    """
    Graph : un graphe
    return : une couverture trouvée par l'algorithme de branch and bound et le nombre de noeuds visités
    """
    stack = LifoQueue()
    stack.put((Graph.copy() , []))
    couverture = algo_couplage(Graph)
    Bsup = len(couverture)
    nb_noeuds_visites = 0
    while(not stack.empty()) :
        nb_noeuds_visites += 1
        G , c = stack.get()
        if G.number_of_edges() == 0 :
            if len(c) < Bsup:
                Bsup = len(c)
                couverture = c
        else :
            M = algo_couplage(G)
            Binf = len(c) + max(bornes(G , M))
            if Bsup > len(c) + len(M) :
                c2 = c.copy()
                c2.extend(M)
                couverture = c2
                Bsup = len(c) + len(M)
            if Binf > Bsup:
                continue
            else :
                edge = list(G.edges)[0]

                fils_Droit = delete_node (G , edge[0])
                c_Droit = c.copy()
                c_Droit.append(edge[0])
                stack.put((fils_Droit , c_Droit))

                fils_Gauche = delete_node (G , edge[1])
                c_Gauche = c.copy()
                c_Gauche.append(edge[1])
                stack.put((fils_Gauche , c_Gauche))
    return couverture , nb_noeuds_visites

def branchement_bornes_glouton(Graph) :
    """
    Graph : un graphe
    return : une couverture trouvée par l'algorithme de branch and bound en utilisant gouton pour trouver la solution réalisable ainsi que le nombre de noeuds visités
    """
    stack = LifoQueue()
    stack.put((Graph.copy() , []))
    couverture = algo_glouton(Graph)
    Bsup = len(couverture)
    nb_noeuds_visites = 0
    while(not stack.empty()) :
        nb_noeuds_visites += 1
        G , c = stack.get()
        if G.number_of_edges() == 0 :
            if len(c) < Bsup:
                Bsup = len(c)
                couverture = c
        else :
            M = algo_couplage(G)
            Binf = len(c) + max(bornes(G , M))
            if Bsup > len(c) + len(M) :
                c2 = c.copy()
                c2.extend(M)
                couverture = c2
                Bsup = len(c) + len(M)
            if Binf > Bsup:
                continue
            else :
                edge = list(G.edges)[0]

                fils_Droit = delete_node (G , edge[0])
                c_Droit = c.copy()
                c_Droit.append(edge[0])
                stack.put((fils_Droit , c_Droit))

                fils_Gauche = delete_node (G , edge[1])
                c_Gauche = c.copy()
                c_Gauche.append(edge[1])
                stack.put((fils_Gauche , c_Gauche))
    return couverture , nb_noeuds_visites



###################################################################################
###                   Partie 4.3 : Amélioration du branchement                  ###    
###################################################################################

# 4.3.1
def branchement_bornes_amelioration_1(Graph) :
    """
    Graph : un graphe
    return : une couverture trouvée par l'algorithme de branch and bound amélioré ainsi que le nombre de noeuds visités

    L'amélioration faite :
        Lorsque l’on branche sur une arête e = {u, v}, dans la deuxième branche où l’on prend le sommet
        v dans la couverture, on peut supposer que l’on ne prend pas le sommet u (le cas où on le prend étant
        traité dans la première branche. Dans la 2ème branche, ne prenant pas u dans la couverture on doit
        alors prendre tous les voisins de u (et on peut les supprimer du graphe)
    """
    stack = LifoQueue()
    stack.put((Graph.copy() , []))
    couverture = algo_couplage(Graph)
    Bsup = len(couverture)
    nb_noeuds_visites = 0
    while(not stack.empty()) :
        nb_noeuds_visites+=1
        G , c = stack.get()
        if G.number_of_edges() == 0 :
            if len(c) < Bsup:
                Bsup = len(c)
                couverture = c
        else :
            M = algo_couplage(G)
            Binf = len(c) + max(bornes(G , M))
            if Bsup > len(c) + len(M) :
                c2 = c.copy()
                c2.extend(M)
                couverture = c2
                Bsup = len(c) + len(M)
            if Binf > Bsup:
                continue
            else :
                edge = list(G.edges)[0]
                u_neighbors = list(G.neighbors(edge[1]))
                u_neighbors.remove(edge[0])

                fils_Droit = delete_nodes ( G , [edge[0]] + [edge[1]] + u_neighbors ) #on supprime du graphe edge[0] edge[1] et tous les voisins de edge[1]
                c_Droit = c.copy()
                c_Droit.extend([edge[0]] + u_neighbors)# an ajoute à la couverture edge[0] et tous les voisins de edge[1]
                stack.put((fils_Droit , c_Droit))

                fils_Gauche = delete_node (G , edge[1])
                c_Gauche = c.copy()
                c_Gauche.append(edge[1])
                stack.put((fils_Gauche , c_Gauche))
    return couverture ,nb_noeuds_visites
# 4.3.2
def branchement_bornes_amelioration_2(Graph) :
    """
    Graph : un graphe
    return : une couverture trouvée par l'algorithme de branch and bound amélioré ainsi que le nombre de noeuds visités

    L'amélioration faite :
        Afin d’éliminer un maximum de sommets dans la deuxième branche, onchoisir le branchement de manière
        à ce que le sommet u soit de degré maximum dans le graphe restant. 
    """
    stack = LifoQueue()
    stack.put((Graph.copy() , []))
    couverture = algo_couplage(Graph)
    Bsup = len(couverture)
    nb_noeuds_visites = 0
    while(not stack.empty()) :
        nb_noeuds_visites +=1
        G , c = stack.get()
        if G.number_of_edges() == 0 :
            if len(c) < Bsup:
                Bsup = len(c)
                couverture = c
        else :
            M = algo_couplage(G)
            Binf = len(c) + max(bornes(G , M))
            if Bsup > len(c) + len(M) :
                c2 = c.copy()
                c2.extend(M)
                couverture = c2
                Bsup = len(c) + len(M)
            if Binf > Bsup:
                continue
            else :
                # on choisi l'arrête qui a le degree max
                max_degree_node = max_degree(G)
                  
                u_neighbors = list(G.neighbors(max_degree_node))
                edge = u_neighbors[0] , max_degree_node
                u_neighbors.remove(edge[0])

                fils_Droit = delete_nodes ( G , [edge[0]] + [edge[1]] + u_neighbors ) #on supprime du graphe edge[0] edge[1] et tous les voisins de edge[1]
                c_Droit = c.copy()
                c_Droit.extend([edge[0]] + u_neighbors)# an ajoute à la couverture edge[0] et tous les voisins de edge[1]
                stack.put((fils_Droit , c_Droit))

                fils_Gauche = delete_node (G , edge[1])
                c_Gauche = c.copy()
                c_Gauche.append(edge[1])
                stack.put((fils_Gauche , c_Gauche))
    return couverture , nb_noeuds_visites

###################################################################################
###                   Qualité des algorithmes approchés                         ###    
###################################################################################

def evaluation_couplage_glouton(nMin , nMax , nb_iter , show = False):

    approx_glouton = []
    approx_couplage = []

    nb_nodes = np.array(range(nMin , nMax+1 ,nMax// nb_iter))

    p =0.5
    
    for n in nb_nodes :
        #On génére un graphe de taille n et de probabilité p
        graph = generate_alea_graphe(n , p)
        while(len(list(graph.edges))== 0):
            graph = generate_alea_graphe(n , p)

        
        couverture_couplage = algo_couplage(graph)
        couverture_glouton = algo_glouton(graph)
        couverture_opt , _ = branchement_bornes_amelioration_2(graph)

        #on calcule le rapport d'approximation
        approx_couplage.append(len(couverture_couplage)/ len(couverture_opt))
        approx_glouton.append(len(couverture_glouton)/ len(couverture_opt))
        

    print("Le pire rapport d'approximation de l'algo de couplage : ", max(approx_couplage))
    print("Le pire rapport d'approximation de l'algo de glouton : ", max(approx_glouton ))

    plt.plot(nb_nodes, approx_couplage, 'b', label="Algo Couplage")
    plt.plot(nb_nodes, approx_glouton, 'g', label="Algo Glouton")
    plt.xlabel('Nombre de sommets')
    plt.ylabel('Rapport d\'approximation')
    plt.legend()
    plt.suptitle("Analyse du rapport d'approximation de l\'agorithme de glouton et de couplage" )
    plt.tight_layout()
    plt.savefig("Plots/Temps_de_calcul/Analyse du rapport d'approximation de l\'agorithme de glouton et de couplage avec n ∈ ["+str(nMin)+","+str(nMax)+"] 2.png")
    
    if show :
        plt.show()

###################################################################################
###                           Fonction auxiliares                               ###    
###################################################################################

def draw_graph(G):
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def import_graph_from_txt(file_name):
    # Créez un graphe vide
    G = nx.Graph()

    with open(file_name, 'r') as file:
        lines = file.readlines()
        nb_nodes = int(lines[1]) 
        nodes = [int(line) for line in lines[3:3+nb_nodes]]
        nb_edges = int(lines[3+nb_nodes+1])
        edges = [tuple(map(int, line.split())) for line in lines[3+nb_nodes+3:]]

        # Ajouter les nœuds au graphe
        G.add_nodes_from(nodes)

        # Ajouter les arêtes au graphe
        G.add_edges_from(edges)

        return G
#####################################################################################################################################################
###                                  Fonction d'analyse de comparaisons des algorithmes de Branch and Bounds                                      ###    
#####################################################################################################################################################
def compare_Branch_and_BranchAndBound(nMin , nMax , nb_iter = 10, nb_mean =10 , show = True ):

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))

    nodes_branch = []
    execution_time_branch = []
    nodes_branchAndBound = []
    execution_time_branchAndBound = []

    nb_nodes = np.array(range(nMin , nMax+1 ,nMax// nb_iter))

    probabilities = [random.random() for i in range(nb_mean)]
    
    for n in nb_nodes :
        
        nodes_branch_i = 0
        execution_time_branch_i = 0
        nodes_branchAndBound_i = 0
        execution_time_branchAndBound_i = 0
        for i in range(nb_mean):
            #On génére un graphe de taille n et de probabilité p
            graph = generate_alea_graphe(n , probabilities[i])
            
            #On calcul le temps d'execution de l'algorithme de branch ainsi que la taille de sa couverture
            time_start = time.time()
            couverture_branch , nb_noeuds_visites_branch= branchment(graph)
            execution_time_branch_i += time.time() - time_start
            nodes_branch_i += nb_noeuds_visites_branch

            #On calcul le temps d'execution de l'algorithme de branchAndBound ainsi que la taille de sa couverture
            time_start = time.time()
            couverture_branchAndBound , nb_noeuds_visites_BB= branchement_bornes(graph)
            execution_time_branchAndBound_i += time.time() - time_start
            nodes_branchAndBound_i += nb_noeuds_visites_BB
        
        #On fait les calcul pour une approximation de nb_mean executions
        nodes_branch.append(nodes_branch_i/nb_mean)
        execution_time_branch.append(execution_time_branch_i/nb_mean)
        nodes_branchAndBound.append(nodes_branchAndBound_i/nb_mean)
        execution_time_branchAndBound.append(execution_time_branchAndBound_i/nb_mean)

    
    axs[0].plot(nb_nodes, execution_time_branch, 'r', label="Algo Branch")
    axs[0].plot(nb_nodes, execution_time_branchAndBound, 'b', label="Algo BranchAndBound")
    axs[0].set_xlabel('Nombre de sommets')
    axs[0].set_ylabel('Temps d\'execution')
    axs[0].legend()

    axs[1].plot(nb_nodes, nodes_branch, 'r', label="Algo Branch")
    axs[1].plot(nb_nodes, nodes_branchAndBound, 'b', label="Algo BranchAndBound")
    axs[1].set_xlabel('Nombre de sommets')
    axs[1].set_ylabel('Nombre de noeuds visités')
    axs[1].legend()

    fig.suptitle("Comparaison du temps d\'execution et le nombre de noeuds visités entre l\'algoritheme de branch et de branchAndBound" )
    plt.tight_layout()
    plt.savefig("Plots/Temps_de_calcul/Comparaison du temps d\'execution et le nombre de noeuds visités entre l\'algorithme de branch et de branchAndBound avec n ∈ ["+str(nMin)+","+str(nMax)+"] 2.png")
    
    if show :
        plt.show()

    
def compare_BranchAndBound_Ameliorations(nMin , nMax , nb_iter = 10, nb_mean = 10 , show = True ):

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))

    nodes_branchAndBound = []
    execution_time_branchAndBound = []
    nodes_branchAndBound_M1 = []
    execution_time_branchAndBound_M1 = []
    nodes_branchAndBound_M2 = []
    execution_time_branchAndBound_M2 = []

    nb_nodes = np.array(range(nMin , nMax+1 ,nMax// nb_iter))

    probabilities = [random.random() for i in range(nb_mean)]
    
    for n in nb_nodes :
        
        nodes_branchAndBound_i = 0
        execution_time_branchAndBound_i = 0
        nodes_branchAndBound_M1_i = 0
        execution_time_branchAndBound_M1_i = 0
        nodes_branchAndBound_M2_i = 0
        execution_time_branchAndBound_M2_i = 0
        for i in range(nb_mean):
            #On génére un graphe de taille n et de probabilité p
            graph = generate_alea_graphe(n , probabilities[i])
            
            #On calcul le temps d'execution de l'algorithme de branchAndBound ainsi que la taille de sa couverture
            time_start = time.time()
            couverture_branchAndBound , nb_noeuds_visites_BB= branchement_bornes(graph)
            execution_time_branchAndBound_i += time.time() - time_start
            nodes_branchAndBound_i += nb_noeuds_visites_BB

            time_start = time.time()
            couverture_branchAndBound_M1 , nb_noeuds_visites_BB_M1= branchement_bornes_amelioration_1(graph)
            execution_time_branchAndBound_M1_i += time.time() - time_start
            nodes_branchAndBound_M1_i += nb_noeuds_visites_BB_M1

            time_start = time.time()
            couverture_branchAndBound_M2 , nb_noeuds_visites_BB_M2= branchement_bornes_amelioration_2(graph)
            execution_time_branchAndBound_M2_i += time.time() - time_start
            nodes_branchAndBound_M2_i += nb_noeuds_visites_BB_M2
        
        #On fait les calcul pour une approximation de nb_mean executions
        nodes_branchAndBound.append(nodes_branchAndBound_i/nb_mean)
        execution_time_branchAndBound.append(execution_time_branchAndBound_i/nb_mean)
        nodes_branchAndBound_M1.append(nodes_branchAndBound_M1_i/nb_mean)
        execution_time_branchAndBound_M1.append(execution_time_branchAndBound_M1_i/nb_mean)
        nodes_branchAndBound_M2.append(nodes_branchAndBound_M2_i/nb_mean)
        execution_time_branchAndBound_M2.append(execution_time_branchAndBound_M2_i/nb_mean)

   
    axs[0].plot(nb_nodes, execution_time_branchAndBound, 'b', label="Algo BranchAndBound")
    axs[0].plot(nb_nodes, execution_time_branchAndBound_M1, 'r', label="Algo BranchAndBound Amelioration 1")
    axs[0].plot(nb_nodes, execution_time_branchAndBound_M2, 'g', label="Algo BranchAndBoundavec glouton")
    axs[0].set_xlabel('Nombre de sommets')
    axs[0].set_ylabel('Temps d\'execution')
    axs[0].legend()

    axs[1].plot(nb_nodes, nodes_branchAndBound, 'b', label="Algo BranchAndBound")
    axs[1].plot(nb_nodes, nodes_branchAndBound_M1, 'r', label="Algo BranchAndBound Amelioration 1")
    axs[1].plot(nb_nodes, nodes_branchAndBound_M2, 'g', label="Algo BranchAndBound avec glouton")
    axs[1].set_xlabel('Nombre de sommets')
    axs[1].set_ylabel('Nombre de noeuds visités')
    axs[1].legend()

    fig.suptitle("Comparaison du temps d\'execution et le nombre de noeuds visités entre les algoritheme améliorations de branchAndBound" )
    plt.tight_layout()
    plt.savefig("Plots/Temps_de_calcul/Comparaison du temps d\'execution et le nombre de noeuds visités entre les algoritheme améliorations de branchAndBound avec n ∈ ["+str(nMin)+","+str(nMax)+"] 2.png")
    
    if show :
        plt.show()

###################################################################################
###                                   Main                                      ###    
###################################################################################

G = import_graph_from_txt("instance1.txt")


couverture_couplage = algo_couplage(G)
couverture_glouton = algo_glouton(G)
couverture_branchement , nb_nodes_branch = branchment(G)
couverture_BandB , nb_nodes_bBandB = branchement_bornes(G)
couverture_BandB_M1 , nb_nodes_bBandB_M1 = branchement_bornes_amelioration_1(G)
couverture_BandB_M2 , nb_nodes_bBandB_M2 = branchement_bornes_amelioration_2(G)

print("La couverture de ce graphe obtenu avec l'algorithme de couplage est " , couverture_couplage)
print("La couverture de ce graphe obtenu avec l'algorithme de glouton est " , couverture_glouton)
print("La couverture de ce graphe obtenu avec l'algorithme de branchement est " , couverture_branchement , ", avec : ",nb_nodes_branch ," noeuds visités")
print("La couverture de ce graphe obtenu avec l'algorithme de branch and bound est " , couverture_BandB , ", avec : ",nb_nodes_bBandB ," noeuds visités")
#print("La couverture de ce graphe obtenu avec l'algorithme de l'amélioration de branch and bound 1 est " , couverture_BandB_M1 , ", avec : ",nb_nodes_bBandB_M1 ," noeuds visités")
#print("La couverture de ce graphe obtenu avec l'algorithme de l'amélioration de branch and bound 2 est " , couverture_BandB_M2 , ", avec : ",nb_nodes_bBandB_M2 ," noeuds visités")

draw_graph(G)

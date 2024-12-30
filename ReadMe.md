# Multi-dimensional multi-option opinion dynamics leads to the emergence of clusters in social networks

In real-world social networks, opinions evolve within a multidimensional space of multiple topics being concurrently discussed, and a multi-option decisionmaking process, rather than a simple binary choice, takes place. Our work introduces a multi-dimensional multi-option opinion dynamics model capturing thecomplexity of opinion evolution in social networks. The model exploits the coupling of inner opinion and outward action, emphasizing how similar actions strengthen interactions between agents. Unlike existing research, in which consensus, clustering or polarization result from specific network structures, we find that different attitude patterns towards neighbours lead to the spontaneous emergence of such macroscopic phenomena, which are therefore independent of network structural features. We provide analytical conditions for the transitions to these behaviors, confirming them via simulations on different networks. Thus, our model allows one to explain the emergence of collective phenomena observed in real-world situations, thereby providing insights in areas such as opinion guidance and multi-agent decision-making.

## Code Instructions
The main code is "Main.py". 

By setting the graph generation in the code "G = nx.barabasi_albert_graph" to SF or "nx.random_graphs.watts_strogatz_graph" to WS network, the results of Figs. 1-4 (corresponding to SF network) and Figs. 5-8 (corresponding to WS network) are obtained respectively. 

## Generated Data
The generated data is saved as a ".mat" file through the "Save_results.ipynb".
The data can be retrieved through the following link:
https://www.dropbox.com/scl/fi/t3ug0h4e27odstn34vrg0/Data-of-OpDyn.zip?rlkey=kbaju9d4lz42kr2dmg33ibx5y&st=f0taxhov&dl=0

All the figures are drawn in MATLAB. See the folder "Draw_Figures".


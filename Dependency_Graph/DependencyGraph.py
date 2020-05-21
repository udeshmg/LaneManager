import networkx as nx
from collections import OrderedDict
import matplotlib.pyplot as plt
import time

class DependencyGraph():

    def __init__(self):
        self.diG = nx.DiGraph()
        self.len = 5
        self.max_exec_time = 0

    def buildGraph(self, OD_list):
        for path in OD_list:
            print(path["path"], path["path"][0], path["path"][1])

    def createVariableDAG(self,G,OD_list):
        #self.diG = nx.DiGraph()

        for k in OD_list:
            i = k["path"]
            u = G[i[0]][i[1]]['edgeId']
            u1 = (i[0], i[1])

            for j in range(min(len(i) - 1, 15)):

                # v = getIndex(i[k],i[k+1], gridSize)
                v = G[i[j]][i[j + 1]]['edgeId']
                v1 = (i[j], i[j + 1])
                if self.getDirectionRelationship(u1,v1):
                    if self.diG.has_edge(u, v) == True:
                        if self.diG[u][v]['temp_direction'] == -1:
                            self.diG[u][v]['temp_direction'] = 0
                        else:
                            self.diG[u][v]['temp_direction'] += 1
                    else:
                        self.diG.add_weighted_edges_from([(u, v, 1)], weight='temp_direction')
                        if i[0] > i[1]:
                            self.diG[u][v]['connect_from'] = 'up_direction'
                        else:
                            self.diG[u][v]['connect_from'] = 'down_direction'
                else:
                    if self.diG.has_edge(u, v) == True:
                        if self.diG[u][v]['temp_direction'] == 1:
                            self.diG[u][v]['temp_direction'] = 0
                        else:
                            self.diG[u][v]['temp_direction'] = self.diG[u][v]['temp_direction']  - 1
                    else:
                        self.diG.add_weighted_edges_from([(u, v, -1)], weight='temp_direction')
                        if i[0] > i[1]:
                            self.diG[u][v]['connect_from'] = 'up_direction'
                        else:
                            self.diG[u][v]['connect_from'] = 'down_direction'

        alpha = 0.9
        for u,v in self.diG.edges():
            if 'direction' in self.diG[u][v]:
                self.diG[u][v]['direction'] = alpha*self.diG[u][v]['direction']+(1-alpha)*self.diG[u][v]['temp_direction']
                self.diG[u][v]['temp_direction'] = 0
            else:
                self.diG[u][v]['direction'] = self.diG[u][v]['temp_direction']

        pass

    def getDirectionRelationship(self,edge1, edge2):
        direc1 = self.find_direction(edge1[0], edge1[1])
        direc2 = self.find_direction(edge2[0], edge2[1])

        if direc1 == direc2:
            return True
        else:
            return False

    def find_direction(self,u, v):
        if (u > v):
            return 'UP'
        else:
            return 'DOWN'

    def assignLoadToNodesFromRoads(self, G):

        for u,v in G.edges():

            if self.diG.has_node(G[u][v]["edgeId"]):
                if G.nodes[u]["index"] > G.nodes[v]["index"]:
                    self.diG.nodes[G[u][v]["edgeId"]]['imbalance'] = G[u][v]["imbalance"]
                    self.diG.nodes[G[u][v]["edgeId"]]['up_configuration'] = G[u][v]["numLanes"]
                    self.diG.nodes[G[u][v]["edgeId"]]['UP'] = G[u][v]["numVehiclesMvg"]
                else:
                    a = G[u][v]["numVehiclesMvg"]
                    self.diG.nodes[G[u][v]["edgeId"]]['DOWN'] = a
                    self.diG.nodes[G[u][v]["edgeId"]]['down_configuration'] = G[u][v]["numLanes"]

    def assignLoadToNodes(self, roadList):

        for n in self.diG.nodes():
            self.diG.nodes[n]['imbalance'] = roadList[n-1].get_traffic_imbalance(roadList[n-1].upstream_id)  # Current imbalance
            self.diG.nodes[n]['configuration'] = roadList[n-1].get_in_lanes_num(roadList[n-1].upstream_id)  # Current config
            self.diG.nodes[n]['UP'] = roadList[n-1].upTraffic
            self.diG.nodes[n]['DOWN'] = roadList[n-1].downTraffic
            self.diG.nodes[n]['load'] = roadList[n-1].get_num_vehicles(roadList[n-1].upstream_id, 'T') + roadList[n-1].get_num_vehicles(roadList[n-1].downstream_id, 'T')
        return self.diG

    def find_dependency1(self, startNodelist, actionList, depth, threshold, load_th=20):
        conflict_counter = OrderedDict()
        additional_change = []

        for node in self.diG.nodes():
            self.diG.nodes[node]['list'] = []
            self.diG.nodes[node]['change'] = 0
            self.diG.nodes[node]['visited'] = False
            self.diG.nodes[node]['action'] = 0
            self.diG.nodes[node]['depth'] = 0

        startTime = time.time()

        queue = []
        for i in range(len(startNodelist)):
            conflict_counter[startNodelist[i]] = 0
            currentNode = startNodelist[i]
            if self.diG.has_node(currentNode):
                self.diG.nodes[currentNode]['action'] = actionList[i]
                for j in self.diG.successors(currentNode):
                    if ( (self.diG.nodes[currentNode]['action'] == 1 and self.diG[currentNode][j]['connect_from'] == 'up_direction')):

                        if abs(self.diG[currentNode][j]['direction']) > 1:
                            self.diG.nodes[j]['list'].append ( [currentNode, self.diG.nodes[currentNode]['action']*
                                                                self.diG[currentNode][j]['direction'], self.diG.nodes[currentNode]['up_configuration']])
                            if not j in queue:
                                queue.append(j)

                    if ((self.diG.nodes[currentNode]['action'] == -1 and self.diG[currentNode][j]['connect_from'] == 'down_direction')):

                        if abs(self.diG[currentNode][j]['direction']) > 1:
                            self.diG.nodes[j]['list'].append([currentNode, self.diG.nodes[currentNode]['action'] *
                                                              self.diG[currentNode][j]['direction'], self.diG.nodes[currentNode]['down_configuration']])
                            if not j in queue:
                                queue.append(j)
                 ####### Additional code
                for j in self.diG.predecessors(currentNode):
                    if ( (self.diG.nodes[currentNode]['action'] == 1 and self.diG[j][currentNode]['connect_from'] == 'down_direction')):
                        if not currentNode in [row[0] for row in self.diG.nodes[j]['list']]:
                            if abs(self.diG[j][currentNode]['direction']) > 1:
                                self.diG.nodes[j]['list'].append ( [currentNode, self.diG.nodes[currentNode]['action']*
                                                                self.diG[j][currentNode]['direction']*0.1, self.diG.nodes[currentNode]['up_configuration']])

                                if not j in queue:
                                    queue.append(j)

                    if ((self.diG.nodes[currentNode]['action'] == -1 and self.diG[j][currentNode]['connect_from'] == 'up_direction')):
                        if not currentNode in [row[0] for row in self.diG.nodes[j]['list']]:
                            if abs(self.diG[j][currentNode]['direction']) > 1:
                                self.diG.nodes[j]['list'].append([currentNode, self.diG.nodes[currentNode]['action'] *
                                                              self.diG[j][currentNode]['direction']*0.1, self.diG.nodes[currentNode]['down_configuration']])

                                if not j in queue:
                                    queue.append(j)

            else:
                conflict_counter[currentNode] = 0

        for i in queue:
            upstream = 0
            downstream = 0

            self.diG.nodes[i]['list'].sort(key=lambda x: x[1])
            for flow in self.diG.nodes[i]['list']:
                if flow[1] > 0:
                    upstream += flow[1]
                else:
                    downstream += (-flow[1])

            # take decision
            action, increase, decrease = self.takeDecision(upstream, downstream, self.diG.nodes[i]['up_configuration'], self.diG.nodes[i]['down_configuration'], 0, self.diG.nodes[i]['UP'], self.diG.nodes[i]['DOWN'], i)

            if not increase:
                for flow in self.diG.nodes[i]['list']:
                    if flow[1] > 0 and flow[2] >= self.diG.nodes[i]['up_configuration']:
                        conflict_counter[flow[0]] += 1

            if not decrease:
                for flow in self.diG.nodes[i]['list']:
                    if flow[1] < 0 and flow[2] >= self.diG.nodes[i]['down_configuration']:
                        conflict_counter[flow[0]] += 1



            if action != 0:
                additional_change.append([i, action])

        allowed_additional_change = []
        for road,action in additional_change:
            source_allowed = False
            for node, flow, _ in self.diG.nodes[road]['list']:
                if action == 1:
                    if flow > 0:
                        if conflict_counter[node] == 0:
                            source_allowed = True

                if action == -1:
                    if flow < 0:
                        if conflict_counter[node] == 0:
                            source_allowed = True
            if source_allowed:
                allowed_additional_change.append([road, action])

        endTime = time.time()


        if (self.max_exec_time <  endTime-startTime):
            self.max_exec_time =  endTime-startTime

        print("Execution time: ", endTime-startTime, self.max_exec_time)

        return conflict_counter, allowed_additional_change

    def takeDecision(self, upstream_direct, downstream_direct, up_configuration, down_configuration, imbalance, up_direct, down_direct, id):
        action = 0
        decrease = False
        increase = False

        upstream = upstream_direct/up_configuration
        downstream = downstream_direct/down_configuration
        up = up_direct/up_configuration
        down= down_direct/down_configuration

        if (upstream / max(0.1,downstream)) > 1.2 and upstream > 0:
            #if upstream > down:
            #    action = 1
            #    increase = True
            #if down > 0.8*up and down > 5:
            if down > 0.8*upstream and down > 5:
                action = 0
            else:
                action = 1
                increase = True

        elif (downstream / max(0.1,upstream)) > 1.2 and downstream > 0:
            #if downstream > up:
            #    action = -1
            #    decrease = True
            #if up > 0.8*down and up > 5:
            if up > 0.8*downstream and up > 5:
                action = 0
            else:
                action = -1
                decrease = True

        return action, increase, decrease


    def generateCoordinateActions(self, road_network, dependency_in_edges, dependency_in_actions):
        edgeIndexes = []
        self.assignLoadToNodesFromRoads(road_network.osmGraph.nxGraph)
        conflict_counter, additional_changes = self.find_dependency1(dependency_in_edges,
                                                                     dependency_in_actions, 0, 12)

        for conf, index, action in zip(conflict_counter, dependency_in_edges, dependency_in_actions):
            if conflict_counter[conf] < 1:
                if action == 1:  # increase in upstream direction
                    nodes = road_network.osmGraph.indexMap[index]  # node 0 -> 1 points downstream direction
                    edgeIndexes.append(road_network.osmGraph.nxGraph[nodes[1]][nodes[0]][
                                           'index'])  # index of direction for lane reduction
                elif action == -1:  # increase in downstream direction
                    nodes = road_network.osmGraph.indexMap[index]  # node 0 -> 1 points downstream direction
                    edgeIndexes.append(road_network.osmGraph.nxGraph[nodes[0]][nodes[1]][
                                           'index'])  # index of direction for lane reduction

        for road in additional_changes:
            if road[1] == 1:
                nodes = road_network.osmGraph.indexMap[road[0]]
                edgeIndexes.append(road_network.osmGraph.nxGraph[nodes[1]][nodes[0]]['index'])
            elif road[1] == -1:
                nodes = road_network.osmGraph.indexMap[road[0]]
                edgeIndexes.append(road_network.osmGraph.nxGraph[nodes[0]][nodes[1]]['index'])

        return edgeIndexes

    def find_dependency(self, startNodelist, actionList, depth, threshold, load_th=20):
        print("Entered")

        '''temp = []
        print(startNodelist)
        for i in range(len(startNodelist)):
            if not self.diG.has_node(startNodelist[i]):
                temp.append(i)

        for i in range(len(temp)):
            print(temp[i])
            del startNodelist[temp[i]]'''

        for node in self.diG.nodes():
            self.diG.nodes[node]['list'] = []
            self.diG.nodes[node]['change'] = 0
            self.diG.nodes[node]['visited'] = False
            self.diG.nodes[node]['action'] = 0
            self.diG.nodes[node]['depth'] = 0

        additional_changes = []
        conflict_counter = OrderedDict()
        queue_lvl1 = []
        queue_lvl2 = []
        for i in range(len(startNodelist)):
            conflict = 0
            print(startNodelist)
            self.diG.nodes[startNodelist[i]]['list'] = [[startNodelist[i], depth, actionList[i], conflict]]
            self.diG.nodes[startNodelist[i]]['change'] = actionList[i]
            self.diG.nodes[startNodelist[i]]['action'] = actionList[i]
            queue_lvl2.append([])
            queue_lvl1.append([startNodelist[i]])
            conflict_counter[startNodelist[i]] = 0

        while depth != 0:
            for k in range(len(startNodelist)):
                print("### start iteration of ", k)
                while queue_lvl1[k]:
                    print(queue_lvl1)
                    currentNode = queue_lvl1[k].pop(0)

                    increase_action = 0
                    decrease_action = 0
                    if not self.diG.nodes[currentNode]['visited']:
                        print("Current node [unvisited]", currentNode)
                        self.diG.nodes[currentNode]['visited'] = True
                        self.diG.nodes[currentNode]['depth'] = depth

                        increase_action = 0
                        decrease_action = 0
                        for i in range(len(self.diG.nodes[currentNode]['list'])):
                            if self.diG.nodes[currentNode]['list'][i][2] == 1:
                                increase_action += 1
                            elif self.diG.nodes[currentNode]['list'][i][2] == -1:
                                decrease_action += 1
                        #print(" Data ", self.diG.nodes[currentNode]['imbalance'], self.diG.nodes[currentNode]['configuration'], increase_action, decrease_action, self.diG.nodes[currentNode]['load'])
                        print("First Rid: ", currentNode, self.diG.nodes[currentNode]['imbalance'], self.diG.nodes[currentNode]['configuration'],
                              increase_action, decrease_action, self.diG.nodes[currentNode]['load'])
                        inc, dec, action = self.road_decision(self.diG.nodes[currentNode]['imbalance'],
                                                         self.diG.nodes[currentNode]['configuration'],
                                                         increase_action, decrease_action,
                                                         self.diG.nodes[currentNode]['load'], load_th,
                                                              self.diG.nodes[currentNode]['UP'], self.diG.nodes[currentNode]['DOWN'])
                        '''if action != 0:
                            if not currentNode in startNodelist:
                                print("Additional change", currentNode, action)
                                additional_changes.append([currentNode, action])'''

                        self.diG.nodes[currentNode]['action'] = action
                        self.diG.nodes[currentNode]['increase'] = inc
                        self.diG.nodes[currentNode]['decrease'] = dec

                        for i in range(len(self.diG.nodes[currentNode]['list'])):
                            if self.diG.nodes[currentNode]['list'][i][2] == 1:
                                if inc == False:
                                    self.diG.nodes[currentNode]['list'][i][3] += 1
                                    conflict_counter[self.diG.nodes[currentNode]['list'][i][0]] += 1
                            elif self.diG.nodes[currentNode]['list'][i][2] == -1:
                                if dec == False:
                                    self.diG.nodes[currentNode]['list'][i][3] += 1
                                    conflict_counter[self.diG.nodes[currentNode]['list'][i][0]] += 1
                            # compare

                    id = self.findStartNodeIndex(self.diG.nodes[currentNode]['list'], startNodelist[k])

                    for i in self.diG.successors(currentNode):
                        '''if self.diG.nodes[currentNode]['action'] == 0:
                            if increase_action > decrease_action:
                                outcome = self.diG[currentNode][i]['direction'] * (1)
                            elif decrease_action > decrease_action:
                                outcome = self.diG[currentNode][i]['direction'] * (-1)
                            else:
                                outcome = 0'''
                        #else:
                        outcome = self.diG[currentNode][i]['direction'] * self.diG.nodes[currentNode]['action']
                        # print("Child added ", i)
                        index = self.findStartNodeIndex(self.diG.nodes[i]['list'], startNodelist[k])
                        if self.diG.nodes[i]['visited']:
                            if (outcome == 1 and self.diG.nodes[i]['increase'] == False) or (
                                    outcome == -1 and self.diG.nodes[i]['decrease'] == False):
                                if index == -1:
                                    self.diG.nodes[i]['list'].append([startNodelist[k], depth - 1, outcome, 1])
                                    conflict_counter[startNodelist[k]] += 1
                                    queue_lvl2[k].append(i)
                            else:
                                if index == -1:
                                    self.diG.nodes[i]['list'].append([startNodelist[k], depth - 1, outcome, 0])
                                    queue_lvl2[k].append(i)
                        else:
                            if index == -1:
                                self.diG.nodes[i]['list'].append([startNodelist[k], depth - 1, outcome, 0])
                                queue_lvl2[k].append(i)

                queue_lvl1[k] = queue_lvl2[k].copy()
                queue_lvl2[k].clear()
                print("### end iteration of ", k)
            depth -= 1

        for node in self.diG.nodes():
            increase_action = 0
            decrease_action = 0
            for i in range(len(self.diG.nodes[node]['list'])):
                if (conflict_counter[self.diG.nodes[node]['list'][i][0]] < threshold) and self.diG.nodes[node]['list'][i][1] == \
                        self.diG.nodes[node]['depth']:
                    if self.diG.nodes[node]['list'][i][2] == 1:
                        increase_action += 1
                    elif self.diG.nodes[node]['list'][i][2] == -1:
                        decrease_action += 1

            if not (increase_action == 0 and decrease_action == 0):
                print("Rid: ", node, self.diG.nodes[node]['imbalance'], self.diG.nodes[node]['configuration'], increase_action, decrease_action)
                inc, dec, action = self.road_decision(self.diG.nodes[node]['imbalance'],
                                                 self.diG.nodes[node]['configuration'],
                                                 increase_action, decrease_action, self.diG.nodes[node]['load'], load_th,
                                                      self.diG.nodes[node]['UP'], self.diG.nodes[node]['DOWN'])

                self.diG.nodes[node]['action'] = action

                if action != 0:
                    if not node in startNodelist:
                        print("Additional change", node, action)
                        additional_changes.append([node, action])
        print("Conflict Counter", conflict_counter)
        return conflict_counter, additional_changes

    @staticmethod
    def road_decision(imbalance, configuration, increase_action, decrease_action, load=0, thresh=20, up=0, down=0):
        # imbalance 1 means output high
        # conf input lanes
        increase_result = False
        decrease_result = False

        action = 0
        if (imbalance == 1 and configuration == 3):
            increase_result = True
            decrease_result = False
            print("Extra Change")
            action = -1

        elif (imbalance == 2 and configuration == 3):
            increase_result = False
            decrease_result = True
            print("Extra Change")
            action = 1


        elif imbalance == 1:
            if decrease_action > 0 and increase_action == 0:  # when only output increases
                decrease_result = True
                #if load >= thresh:
                action = -1

            elif decrease_action >= 0 and increase_action > 0:
                decrease_result = False
                increase_result = True



        elif (imbalance == 2 and configuration == 4):
            increase_result = False
            decrease_result = True

        elif imbalance == 2:
            if increase_action > 0 and decrease_action == 0:
                increase_result = True
                #if load >= thresh:
                action = 1

            elif increase_action >= 0 and decrease_action > 0:
                decrease_result = True
                increase_result = False




        elif imbalance == 0:
            if configuration == 2:
                if increase_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = False
                    increase_result = True
                    if (0.5*up) > down:
                        action = 1
                    #else:
                    #    print("Almost full: ")

                if up > 100 and down > 100:
                    print("balancing")
                    action = 1

            elif configuration == 4:
                if decrease_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = True
                    increase_result = False
                    if (0.5*down) > up:
                        action = -1
                    #else:
                    #    print("Almost full: ")

                if up > 100 and down > 100:
                    print("balancing")
                    action = -1

            else:  # action change incident
                if increase_action > decrease_action and 0.5*up > down:
                    action = 1
                elif increase_action < decrease_action and 0.5*down > up:
                    action = -1
                decrease_result = True
                increase_result = True
        return increase_result, decrease_result, action

    @staticmethod
    def road_decision1(imbalance, configuration, increase_action, decrease_action, load=0, thresh=20, up=0, down=0):
        # imbalance 1 means output high
        # conf input lanes
        increase_result = False
        decrease_result = False
        thresh1 = 10

        action = 0
        if (imbalance == 1 and configuration == 2):
            increase_result = True
            decrease_result = False

        elif imbalance == 1:

            if decrease_action > 0 and increase_action == 0:  # when only output increases
                decrease_result = True
                if load >= thresh:
                    action = -1

            elif decrease_action >= 0 and increase_action > 0:
                decrease_result = False
                increase_result = True


        elif (imbalance == 2 and configuration == 4):
            increase_result = False
            decrease_result = True

        elif imbalance == 2:

            if increase_action > 0 and decrease_action == 0:
                increase_result = True
                if load >= thresh:
                    action = 1


            elif increase_action >= 0 and decrease_action > 0:
                decrease_result = True
                increase_result = False


        elif imbalance == 0:
            if configuration == 2:
                if increase_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = False
                    increase_result = True
                    if load <= thresh1:
                        action = 1

            elif configuration == 4:
                if decrease_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = True
                    increase_result = False
                    if load <= thresh1:
                        action = -1

            else:  # action change incident
                '''if increase_action > decrease_action and load >= thresh:
                    action = 1
                elif increase_action < decrease_action and load >= thresh:
                    action = -1'''
                decrease_result = True
                increase_result = True
        return increase_result, decrease_result, action

    @staticmethod
    def road_decision2(imbalance, configuration, increase_action, decrease_action, load=0, thresh=20, up=0, down=0):
        # imbalance 1 means output high
        # conf input lanes
        increase_result = False
        decrease_result = False

        action = 0
        if (imbalance == 1 and configuration == 3):
            increase_result = True
            decrease_result = False
            print("Extra Change")
            action = -1

        elif (imbalance == 2 and configuration == 3):
            increase_result = False
            decrease_result = True
            print("Extra Change")
            action = 1


        elif imbalance == 1:
            if decrease_action > 0 and increase_action == 0 and up < 20:  # when only output increases
                decrease_result = True
                if load >= thresh:
                    action = -1

            elif decrease_action >= 0 and increase_action > 0:
                decrease_result = False
                increase_result = True



        elif (imbalance == 2 and configuration == 4):
            increase_result = False
            decrease_result = True

        elif imbalance == 2:
            if increase_action > 0 and decrease_action == 0 and down <20:
                increase_result = True
                if load >= thresh:
                    action = 1

            elif increase_action >= 0 and decrease_action > 0:
                decrease_result = True
                increase_result = False




        elif imbalance == 0:
            if configuration == 2:
                if increase_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = False
                    increase_result = True
                    if down < 20:
                        action = 1
                    #else:
                    #    print("Almost full: ")

                if up > 100 and down > 100:
                    print("balancing")
                    action = 1

            elif configuration == 4:
                if decrease_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = True
                    increase_result = False
                    if up < 20:
                        action = -1
                    #else:
                    #    print("Almost full: ")

                if up > 100 and down > 100:
                    print("balancing")
                    action = -1

            else:  # action change incident
                '''if increase_action > decrease_action and load > thresh:
                    action = 1
                elif increase_action < decrease_action and load > thresh:
                    action = -1'''
                decrease_result = True
                increase_result = True
        return increase_result, decrease_result, action

    def drawGraph(self,block=True, attr='length'):
        plt.figure('Path dependency Graph')
        #pos = dict((u, (self.diG.nodes[u]['x'], self.nxGraph.nodes[u]['y'])) for u in self.nxGraph.nodes())
        pos = nx.layout.spring_layout(self.diG)

        #value = [self.nxGraph[u][v][attr]/20 for u, v in self.nxGraph.edges()]
        nx.draw(self.diG, pos, with_labels=True, node_size=10)
        #nx.draw_networkx_nodes()
        nx.draw_networkx_nodes(self.diG, pos)
        #nx.draw_networkx_edge_labels(self.diG,pos,
        #                             edge_labels=dict([((u, v,), int(d['direction'])) for u, v, d in self.diG.edges(data=True)]))
        nx.draw_networkx_edges(self.diG, pos,  arrowstyle='->',
                                       arrowsize=10, width=2)
        nx.draw_networkx_edge_labels(self.diG, pos, label_pos=0.3,
                               edge_labels=dict([((u, v,), int(d['direction'])) for u, v, d in self.diG.edges(data=True)]))
        plt.show()

    @staticmethod
    def findStartNodeIndex(l, startNode, column=0):
        for i in range(len(l)):
            if l[i][column] == startNode:
                return i
        return -1
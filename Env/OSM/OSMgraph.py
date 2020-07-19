import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque

class OsmGraph():

    def __init__(self):
        self.nxGraph = nx.DiGraph()
        self.indexMap = {}
        self.road_index_map = []

    def buildGraphFromDict(self, dictionary):
        commonEdgeIndex = 0
        d_graph = nx.DiGraph()
        for edge in dictionary:
            d_graph.add_edge(edge["startNode"], edge["endNode"])

            if d_graph.has_edge(edge["endNode"], edge["startNode"]):
                d_graph[edge["startNode"]][edge["endNode"]]["edgeId"] = d_graph[edge["endNode"]][edge["startNode"]]["edgeId"]
            else:
                d_graph[edge["startNode"]][edge["endNode"]]["edgeId"] = commonEdgeIndex
                self.indexMap[commonEdgeIndex] = [edge["startNode"], edge["endNode"]]
                commonEdgeIndex += 1

            #d_graph.nodes[edge["startNode"]]["startNode"] = edge["startNode"]
            #d_graph.nodes[edge["endNode"]]["endNode"] = edge["endNode"]

            if d_graph.has_node(edge["startNode"]):
                d_graph.nodes[edge["startNode"]]["x"] = edge["startLon"]
                d_graph.nodes[edge["startNode"]]["y"] = edge["startLat"]
                d_graph.nodes[edge["startNode"]]["index"]= edge["startNode"]

            if d_graph.has_node(edge["endNode"]):
                d_graph.nodes[edge["endNode"]]["x"] = edge["endLon"]
                d_graph.nodes[edge["endNode"]]["y"] = edge["endLat"]
                d_graph.nodes[edge["endNode"]]["index"] = edge["endNode"]

            #assign Edge Atrributes

            for attr in edge:
                d_graph[edge["startNode"]][edge["endNode"]][attr] = edge[attr]

            d_graph[edge["startNode"]][edge["endNode"]]['imbalance'] = 0

            d_graph[edge["startNode"]][edge["endNode"]]["length"] = 12

        print("Road Network : Nodes", d_graph.nodes(data=True))
        print("Road Network : Edges", d_graph.edges(data=True))
        self.nxGraph = d_graph

        return self.indexMap

    def build_edge_map(self):
        for u,v in self.nxGraph.edges():
            if u < v:
                self.road_index_map.append([u, v])

        #####
        print("Number of nodes: ", len(self.nxGraph.nodes()))
        print("Number of edges: ", len(self.nxGraph.edges()))


    def edge_index_from_nodes(self, actions):
        roads = []
        commonIndex = []
        dependency_actions= []

        for index, action in enumerate(actions):
            u, v = self.road_index_map[index]
            # interchange direction:
            # 1. v -> u : upstream ( from higher to lower)
            if action == 0:
                roads.append(self.nxGraph[u][v]["index"])
                commonIndex.append(self.nxGraph[u][v]["edgeId"])
                dependency_actions.append(-1)


            elif action == 2:
                roads.append(self.nxGraph[v][u]["index"])
                commonIndex.append(self.nxGraph[u][v]["edgeId"])
                dependency_actions.append(1)

        return roads, commonIndex, dependency_actions

    def get_road_data(self, index):

        u, v = self.road_index_map[index]
        upstream = int(round(self.nxGraph[v][u]["numVehiclesMvg"]))
        downstream = int(round(self.nxGraph[u][v]["numVehiclesMvg"]))
        lanes = int(round(self.nxGraph[v][u]["numLanes"]))

        return [upstream, downstream, lanes]


    def drawGraph(self,block=True, attr='length'):
        pos = dict((u, (self.nxGraph.nodes[u]['x'], self.nxGraph.nodes[u]['y'])) for u in self.nxGraph.nodes())
        value = [self.nxGraph[u][v][attr]/20 for u, v in self.nxGraph.edges()]
        nx.draw(self.nxGraph, pos, with_labels=True, width=value, node_size=10)
        nx.draw_networkx_nodes(self.nxGraph, pos)
        nx.draw_networkx_edge_labels(self.nxGraph,pos,
                                     edge_labels=dict([((u, v,), int(d[attr])) for u, v, d in self.nxGraph.edges(data=True)]))
        plt.show(block)

    def updateTrafficData(self, dictionary):
        for edge in dictionary:
            for attr in edge:
                self.nxGraph[edge["startNode"]][edge["endNode"]][attr] = edge[attr]

    def filterSDpairs(self):
        for i in range(len(self.SDpairs)):
            if self.SDpairs[i][0] != self.SDpairs[i][1]:
                self.filteredSDpairs.append(self.SDpairs[i])

    def drawGraphWithUserTraffic(self, block=True , figName="Time selected"):
        pos = dict((u, (self.nxGraph.nodes[u]['x'], self.nxGraph.nodes[u]['y'])) for u in self.nxGraph.nodes())
        value = [self.nxGraph[u][v]['path']/10 for u, v in self.nxGraph.edges()]
        #plt.figure(figName)
        colorMap = []
        '''for node in self.nxGraph.nodes():
            if (node in [i[0] for i in self.filteredSDpairs]) and (node in [i[1] for i in self.filteredSDpairs]):
                colorMap.append('b')
            elif (node in [i[0] for i in self.filteredSDpairs]):
                colorMap.append('r')
            elif (node in [i[1] for i in self.filteredSDpairs]):
                colorMap.append('g')
            else:
                colorMap.append('gray')'''

        for node in self.nxGraph.nodes():
            if self.nxGraph.nodes[node]['source'] >  self.nxGraph.nodes[node]['destination'] :
                colorMap.append('r')
            elif self.nxGraph.nodes[node]['source'] < self.nxGraph.nodes[node]['destination']:
                colorMap.append('g')
            elif self.nxGraph.nodes[node]['source'] == 0 and  self.nxGraph.nodes[node]['destination'] == 0:
                colorMap.append('gray')
            else:
                colorMap.append('b')



        nx.draw(self.nxGraph, pos, with_labels=True, width=value, node_size=3)
        nx.draw_networkx_nodes(self.nxGraph, pos, node_color=colorMap)
        nx.draw_networkx_edge_labels(self.nxGraph,pos,
                                     edge_labels=dict([((u, v,), d['edgeId']) for u, v, d in self.nxGraph.edges(data=True)]))
        plt.show()
        #plt.pause(1)

    def drawPathOnMap(self,imbalance=False, type=False, dir='both'):
        for u,v in self.nxGraph.edges():
            self.nxGraph[u][v]['path'] = 0

        for i in range(len(self.SDpaths)):
            for j in range(len(self.SDpaths[i])-1):
                if not imbalance:
                    if type:
                        if dir == 'UP':
                            if self.SDpaths[i][j] > self.SDpaths[i][j+1]:
                                self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j+1]]['path'] += self.SDpairs[i][2]
                        elif dir == 'DOWN':
                            if self.SDpaths[i][j] <= self.SDpaths[i][j+1]:
                                self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j+1]]['path'] += self.SDpairs[i][2]
                        else:
                            self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j + 1]]['path'] += self.SDpairs[i][2]
                    else:
                        self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j+1]]['path'] += 1
                else:
                    if self.nxGraph.nodes[self.SDpaths[i][j]]['id'] > self.nxGraph.nodes[self.SDpaths[i][j+1]]['id']:
                        self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j + 1]]['path'] += self.SDpairs[i][2]

    def getEdgeCoordinates(self, edge):
        x1 = self.nxGraph.nodes[edge[0]]['x']
        y1 = self.nxGraph.nodes[edge[0]]['y']
        x2 = self.nxGraph.nodes[edge[1]]['x']
        y2 = self.nxGraph.nodes[edge[1]]['y']

        return x1,y1,x2,y2

    def getEdgeVector(self, edge):
        x1 = self.nxGraph.nodes[edge[0]]['x']
        y1 = self.nxGraph.nodes[edge[0]]['y']
        x2 = self.nxGraph.nodes[edge[1]]['x']
        y2 = self.nxGraph.nodes[edge[1]]['y']

        return x2-x1, y2-y1

    def getAngleBetweenEdges(self,edgeVector1, edgeVector2):
        #cosine_angle = np.dot(edgeVector1, edgeVector2) / (np.linalg.norm(edgeVector1) * np.linalg.norm(edgeVector2))
        angle = math.atan2(edgeVector2[0],edgeVector2[1]) - math.atan2(edgeVector1[0],edgeVector1[1])
        if angle < 0:
            angle = 2*math.pi - abs(angle)
        return angle

    def createAdjacenceEdgeAttribute(self):
        for u,v in self.nxGraph.edges():

            if self.nxGraph.degree[u] == 4:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle,edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[2][1], 'R'])


            if self.nxGraph.degree[u] == 3:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle, edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                remainAngle = angleEdgeMap[1][0] - angleEdgeMap[0][0]

                if  abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[0][0]) and \
                    abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[1][0]):
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])
                else:
                    if abs(math.pi - angleEdgeMap[0][0]) <= abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'S'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])

                    if abs(math.pi - angleEdgeMap[0][0]) > abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])

            if self.nxGraph.degree[u] == 2:
                edgeList = list(self.nxGraph.edges(u))
                for i in range(len(edgeList)):
                    if edgeList[i] != (u, v):
                        self.nxGraph[u][v]['angle map'].append([edgeList[i], 'S'])

            temp = v
            v = u
            u = temp
            if self.nxGraph.degree[u] == 4:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle,edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[2][1], 'R'])


            if self.nxGraph.degree[u] == 3:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle, edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                remainAngle = angleEdgeMap[1][0] - angleEdgeMap[0][0]

                if  abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[0][0]) and \
                    abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[1][0]):
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])
                else:
                    if abs(math.pi - angleEdgeMap[0][0]) <= abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'S'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])

                    if abs(math.pi - angleEdgeMap[0][0]) > abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])

            if self.nxGraph.degree[u] == 2:
                edgeList = list(self.nxGraph.edges(u))
                for i in range(len(edgeList)):
                    if edgeList[i] != (u, v):
                        self.nxGraph[u][v]['angle map'].append([edgeList[i], 'S'])

    def getRoadsForIntersection(self,n):
        angleEdgeMap = []
        if self.nxGraph.degree[n] == 4:
            edgeList = list(self.nxGraph.edges(n))
            firstEdge = edgeList[0]
            for i in range(len(edgeList)):
                if edgeList[i] != firstEdge:
                    angle = self.getAngleBetweenEdges(self.getEdgeVector((firstEdge[1], n)) ,
                                                      self.getEdgeVector((edgeList[i][1], n)))
                    angleEdgeMap.append([angle,edgeList[i]])

            angleEdgeMap.sort(reverse=True)
            angleEdgeMap.insert(0, [0, firstEdge])
            print([row[1] for row in angleEdgeMap] )
            return [row[1] for row in angleEdgeMap], self.nxGraph.degree[n]

        elif self.nxGraph.degree[n] == 3:
            output = [None]*3
            edgeList = list(self.nxGraph.edges(n))
            firstEdge = edgeList[0]

            newList = [u for u in self.nxGraph[firstEdge[0]][firstEdge[1]]['angle map']
                       if u[0] == edgeList[1] or u[0] == edgeList[2]]
            #print("3: list", newList)

            if 'R' in [row[1] for row in newList] and 'L' in [row[1] for row in newList]:
                output[0] = firstEdge
                output[1] = [row[0] for row in newList if row[1] == 'R'][0]
                output[2] = [row[0] for row in newList if row[1] == 'L'][0]

            elif 'R' in [row[1] for row in newList] and 'S' in [row[1] for row in newList]:
                output[0] = [row[0] for row in newList if row[1] == 'R'][0]
                output[1] = [row[0] for row in newList if row[1] == 'S'][0]
                output[2] = firstEdge

            elif 'L' in [row[1] for row in newList] and 'S' in [row[1] for row in newList]:
                output[0] = [row[0] for row in newList if row[1] == 'L'][0]
                output[1] = firstEdge
                output[2] = [row[0] for row in newList if row[1] == 'S'][0]

            #print(output)
            return output, self.nxGraph.degree[n]

        elif self.nxGraph.degree[n] == 2 or self.nxGraph.degree[n] == 1 :
            return list(self.nxGraph.edges(n)), self.nxGraph.degree[n]

        else:
            return 0, 0



    def getNearestNode(self,locationList):
        SDList =[[-1,-1,-1.0,-1.0, -1] for i in range(len(locationList))]
        for n in self.nxGraph.nodes():
            for l in range(len(locationList)):
                nodeLoc = [self.nxGraph.nodes[n]['x'], self.nxGraph.nodes[n]['y']]
                source = [locationList[l][0],locationList[l][1]]
                destination = [locationList[l][2], locationList[l][3]]

                distance1 = self.calculateDistance2p(source, nodeLoc)
                distance2 = self.calculateDistance2p(destination, nodeLoc)

                #print("Distance: ", distance1, distance2)

                if SDList[l][2] > distance1 or SDList[l][2] == -1:
                    SDList[l][0] = self.nxGraph.nodes[n]['id']
                    SDList[l][2] = distance1

                if SDList[l][3] > distance2 or SDList[l][3] == -1:
                    SDList[l][1] = self.nxGraph.nodes[n]['id']
                    SDList[l][3] = distance2

                SDList[l][4] = locationList[l][4]

        for l in range(len(SDList)):
            self.nxGraph.nodes[SDList[l][0]]['source'] += 1
            self.nxGraph.nodes[SDList[l][1]]['destination'] += 1

            path = nx.shortest_path(self.directedG, SDList[l][0], SDList[l][1], weight='link time')
            #path = nx.shortest_path(self.nxGraph, SDList[l][0], SDList[l][1])
            SDList[l].append(path)
            if path not in self.recentPaths:
                self.recentPaths.append(path)

            if path not in [row[0] for row in self.recentPathsWithDemand]:
                self.recentPathsWithDemand.append([path, 1])
            else:
                for i in range(len(self.recentPathsWithDemand)):
                    if self.recentPathsWithDemand[i][0] == path:
                        demand = self.recentPathsWithDemand[i][1]
                        del self.recentPathsWithDemand[i]
                        self.recentPathsWithDemand.append([path, demand+1])



            if [SDList[l][0],SDList[l][1]] not in self.SDpairs:
                self.SDpairs.append([SDList[l][0],SDList[l][1]])
                self.SDpaths.append(path)

        return [[row[0],row[1],row[5],row[4]] for row in SDList]

    def getPathFromNodes(self, source, destination, load):
        path = nx.shortest_path(self.directedG, source, destination, weight='link time')


        self.nxGraph.nodes[source]['source'] += load
        self.nxGraph.nodes[destination]['destination'] += load
        self.recentPaths.append(path)
        if [source, destination] not in [[row[0],row[1]] for row in self.SDpairs]:
            self.SDpairs.append([source, destination, load])
            self.SDpaths.append(path)

        return source, destination, path

    def allocateLaneBasedOnLoad(self):
        roadChanges = []
        print(self.SDpairs)
        for u, v in self.nxGraph.edges:
            self.nxGraph.edges[u, v]['down'] = 0
            self.nxGraph.edges[u, v]['up'] = 0

        self.SDpairs = self.SDpairs[-20:len(self.SDpairs)]
        self.SDpaths = self.SDpaths[-20:len(self.SDpaths)]

        for i in range(20):
            load = self.SDpairs[i][2]
            for j in range(len(self.SDpaths[i]) - 1):
                if self.nxGraph.nodes[self.SDpaths[i][j]]['id'] < self.nxGraph.nodes[self.SDpaths[i][j + 1]]['id']:
                    self.nxGraph[self.nxGraph.nodes[self.SDpaths[i][j]]['id']][
                        self.nxGraph.nodes[self.SDpaths[i][j + 1]]['id']]['up'] += load
                else:
                    self.nxGraph[self.nxGraph.nodes[self.SDpaths[i][j]]['id']][
                        self.nxGraph.nodes[self.SDpaths[i][j + 1]]['id']]['down'] += load

        for u, v in self.nxGraph.edges:
            imbalance = (self.nxGraph.edges[u, v]['down'] - self.nxGraph.edges[u, v]['up']) / max(
                (self.nxGraph.edges[u, v]['down'] + self.nxGraph.edges[u, v]['up']), 1)

            minLoad = min(self.nxGraph.edges[u, v]['down'], self.nxGraph.edges[u, v]['up'])

            if imbalance > 0.3:
                roadChanges.append([self.nxGraph.edges[u, v]['edgeId'], 1])
            elif imbalance < -0.3:
                roadChanges.append([self.nxGraph.edges[u, v]['edgeId'], 2])

        print(roadChanges)
        return roadChanges

    def allocateLaneBasedOnLoadTime(self, pathList):
        roadChanges = []

        for u, v in self.nxGraph.edges:
            self.nxGraph.edges[u, v]['down'] = 0
            self.nxGraph.edges[u, v]['up'] = 0

        for i in range(len(pathList)):
            for j in range(len(pathList[i])-1):
                val1 = pathList[i][j]
                val2 = pathList[i][j+1]

                if self.nxGraph.nodes[val1]['id'] < self.nxGraph.nodes[val2]['id']:
                    self.nxGraph[val1][val2]['up'] += 1
                else:
                    self.nxGraph[val1][val2]['down'] += 1

        for u, v in self.nxGraph.edges:
            imbalance = (self.nxGraph.edges[u, v]['down'] - self.nxGraph.edges[u, v]['up']) / max(
                (self.nxGraph.edges[u, v]['down'] + self.nxGraph.edges[u, v]['up']), 1)

            minLoad = min(self.nxGraph.edges[u, v]['down'], self.nxGraph.edges[u, v]['up'])

            if minLoad < self.minLoad:
                if imbalance > 0.3:
                    roadChanges.append([self.nxGraph.edges[u, v]['edgeId'], 1])
                elif imbalance < -0.3:
                    roadChanges.append([self.nxGraph.edges[u, v]['edgeId'], 2])


        return roadChanges


    def allocateLaneBasedOnPaths(self):
        roadChanges = []
        for u, v in self.nxGraph.edges:
            self.nxGraph.edges[u, v]['down'] = 0
            self.nxGraph.edges[u, v]['up'] = 0

        #self.SDpairs = self.SDpairs[-20:len(self.SDpairs)]
        #self.SDpaths = self.SDpaths[-20:len(self.SDpaths)]

        for i in range(len(self.recentPathsWithDemand)):
            load = self.recentPathsWithDemand[i][1]
            for j in range(len(self.recentPathsWithDemand[i][0])-1):
                val1 = self.recentPathsWithDemand[i][0][j]
                val2 = self.recentPathsWithDemand[i][0][j+1]

                if self.nxGraph.nodes[val1]['id'] < self.nxGraph.nodes[val2]['id']:
                    self.nxGraph[val1][val2]['up'] += load
                else:
                    self.nxGraph[val1][val2]['down'] += load

        for u, v in self.nxGraph.edges:
            imbalance = (self.nxGraph.edges[u, v]['down'] - self.nxGraph.edges[u, v]['up']) / max(
                (self.nxGraph.edges[u, v]['down'] + self.nxGraph.edges[u, v]['up']), 1)

            minLoad = min(self.nxGraph.edges[u, v]['down'], self.nxGraph.edges[u, v]['up'])

            if minLoad > 25:
                if imbalance > 0.3:
                    roadChanges.append([self.nxGraph.edges[u, v]['edgeId'], 1])
                elif imbalance < -0.3:
                    roadChanges.append([self.nxGraph.edges[u, v]['edgeId'], 2])


        return roadChanges



    @staticmethod
    def calculateDistance2p(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


####### Testing
#nxGraph = OsmGraph(-73.9189, 40.7468, 300)
#a = [[-73.9191, 40.7479, -73.9168, 40.7455]]
#nxGraph.drawGraph()
#print(nxGraph.getNearestNode(a))


'''
self.nxGraph.createAdjacenceEdgeAttribute()
for u,v in self.nxGraph.nxGraph.edges():
    print(u,v)
    print(self.nxGraph.nxGraph[u][v]['angle map'])

                        '''
#spatialG = ox.graph_from_point(( 40.744612, -73.995830), distance=600, network_type='drive')
#ox.plot_graph(spatialG, edge_linewidth=4, edge_color='b')

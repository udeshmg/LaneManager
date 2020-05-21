from Env.OSM.OSMgraph import OsmGraph


class RoadNetwork:

    def __init__(self):
        self.osmGraph = None
        self.agentHandlers = []

    def buildGraph(self):
        print("Loading data from Open street maps...")
        self.osmGraph = OsmGraph()
        print("Map load complete")


    def buildGraphFromDict(self, dictionary):
        self.osmGraph = OsmGraph()
        self.osmGraph.buildGraphFromDict(dictionary)
        #self.osmGraph.drawGraph(True, "edgeId")


    def updateTrafficData(self, dictionary):
        self.osmGraph.updateTrafficData(dictionary)






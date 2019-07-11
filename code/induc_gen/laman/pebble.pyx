"""
This is a pebble game algorithm written in python, translated from the original program in Fortran.
To run this program, you need to have python and cython installed.
By Leyou Zhang, 2016, 04
"""

class lattice(object):

    def __init__(self):
        self.digraph = {} # directed graph
        self.graph = {} # undirected graph (complete)
        self.cluster={} # cluster information
        self.bond = []
        self.stress = []
        self.visited = []
        self.statistics = {}

    def clear(self):
        self.__init__()

    def add_bond(self,int x,int y):

        """
        :param x: a site
        :param y: a different site
        :return: if the new bond is independent: True, otherwise False.
        """

        # no self-loop bonds:
        if x == y:
            raise ValueError('add_bond must have two different sites')

        # skip pebble game if this edge already exists
        # we will be adding bonds from a dict of sets where some bonds are represented twice
        if x in self.graph:
            if y in self.graph[x]:
                return True

        # smaller site first:
        x,y = sorted((x,y))
        # update bonds:
        self.bond.append((x,y))

        # update complete graph
        sites = self.graph.keys()  # potentially remove this
        if x not in sites:
            self.graph[x] = [y]
        else:
            self.graph[x].append(y)
        if y not in sites:
            self.graph[y] = [x]
        else:
            self.graph[y].append(x)
        # update directed graph
        sites = self.digraph.keys()
        if x not in sites:
            self.digraph[x] = [[y],1]
            if y not in sites:
                self.digraph[y] = [[],2]
            return True
        elif y not in sites:
            self.digraph[y] = [[x],1]
            return True
        elif self.collect_four_pebble(x,y):
            if y not in self.digraph[x][0]:
                self.digraph[x][0].append(y)
            try:
                self.digraph[x][1] = self.digraph[x][1] -1
            except Exception as f:
                raise KeyError(f.message)
            return True
        else:
            return False

            # check independence

    def depth_first_search(self,int x,int y, z = False, status = 'start'):

        if status == 'start':
            if not z:
                self.visited = [x,y]
            else:
                self.visited = [x,y,z]
        else:
            self.visited.append(x)

        # exclude y (or y,z) in the search
        if not z:
            if x == y:
                raise ValueError('depth_first_search must have two or three different sites')
            for i in self.digraph[x][0]:
                if i not in self.visited:
                    if self.digraph[i][1] > 0:
                        tree = [i]
                        return tree
                    tree = self.depth_first_search(i,y,status='next')
                    if tree:
                        return [i]+tree
        else:
            if x == y or x == z or y == z:
                raise ValueError('depth_first_search must have two or three different sites')
            for i in self.digraph[x][0]:
                if i not in self.visited:
                    if self.digraph[i][1] > 0:
                        tree = [i]
                        return tree
                    tree = self.depth_first_search(i,y, z = z,status='next')
                    if tree:
                        return [i]+tree
        return None


    def collect_one_pebble(self,int x,int y):

        """
        :param x: a site
        :param y: a different site
        :return: if the one pebble can be collected, return True, otherwise False.
        """

        sites = self.graph.keys()
        if x in sites:
            tree = self.depth_first_search(x,y)
            if tree:
                self.digraph[x][1] += 1
                while tree:
                    site = tree.pop(0)
                    self.digraph[x][0].remove(site)
                    self.digraph[site][0].append(x)
                    x = site
                self.digraph[site][1] += - 1
                return True
            else:
                return False
        else:
            raise ValueError('site %d is not in the lattice.'%x)

    def collect_four_pebble(self, int x, int y):

        """
        :param x: a site
        :param y: a different site
        :return: if the four pebble can be collected, return True, otherwise False.
        """

        if x == y:
            raise ValueError('collect_four_pebble must have two different sites')

        freex = self.digraph[x][1]
        freey = self.digraph[y][1]
        while freex < 2:
            if self.collect_one_pebble(x,y):
                freex += 1
            else:
                break
        while freey < 2:
            if self.collect_one_pebble(y,x):
                freey += 1
            else:
                break
        if freex==2 and freey==2:
            return True
        else:
            return False

    def decompose_into_cluster(self):
        cluster = {}
        bond = self.bond[:]
        index = 0
        while bond:
            index += 1
            i = bond.pop()
            self.collect_four_pebble(i[0],i[1]) # collect 3 pebble (because it can't find the forth pebble)
            cluster[i] = index
            for j in bond[:]:
                check = [self.digraph[k][1]<1 and not self.depth_first_search(k,i[0],z=i[1]) for k in j if k not in i]+[True]
                if all(check):
                    cluster[j] = index
                    bond.remove(j)
        self.cluster['bond'] = cluster.copy()

        cluster_site = {}
        for bond,index in cluster.iteritems():
            for i in bond:
                if i in cluster_site.keys():
                    if index not in cluster_site[i]:
                        cluster_site[i] = cluster_site[i]+(index,)
                else:
                    cluster_site[i] = (index,)

        self.cluster['site'] = cluster_site.copy()
        self.cluster['count'] = index

        cluster_index = {}
        for key,value in cluster.iteritems():
            if value in cluster_index.keys():
                cluster_index[value].append(key)
            else:
                cluster_index[value]=[key]
        self.cluster['index'] = cluster_index.copy()

    def decompose_stress(self):
        bond = self.bond[:]
        bond_single = []
        self.stress = []
        while bond:
            i = bond.pop()
            bond_single.append(i)
            if i in bond:
                self.stress.append(i)
                while i in bond:
                    self.stress.append(i)
                    bond.remove(i)
        bond = bond_single[:]
        while bond:
            i = bond.pop()
            if i[0] in self.digraph[i[1]][0] or i[1] in self.digraph[i[0]][0]:
                continue
            else:
                self.collect_four_pebble(i[0],i[1])
                if self.digraph[i[0]][1] == 2:
                    start = [i[1]]
                else:
                    start = [i[0]]
                visited = [i]
                while start:
                    next = []
                    for j in start:
                        for k in self.digraph[j][0]:
                            if (j,k) not in visited:
                                next.append(k)
                                visited.append((j,k))
                                b = tuple(sorted((j,k)))
                                if b not in self.stress:
                                    self.stress.append(b)
                                if b in bond:
                                    bond.remove(b)
                    if next:
                        if i not in self.stress:
                            self.stress.append(i)
                    start = next[:]
    def stat(self):
        site = len(self.graph.keys())
        bond = len(self.bond)
        floppy_mode = 0
        for value in self.digraph.values():
            floppy_mode += value[1]
        self_stress  =  bond + floppy_mode - 2 * site
        self.statistics = {
            'site':site,
            'bond':bond,
            'floppy_mode':floppy_mode,
            'self_stress':self_stress,
        }
        return True

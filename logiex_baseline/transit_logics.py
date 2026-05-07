"""
LogiEx transit logic grammar and typed formula classes.
Copied from original LogiEx code (LogiEx-main/backend/tasks/transit_logics.py).
Only change: removed hardcoded sys.path.insert.
"""
from parsimonious import Grammar, NodeVisitor
from collections import namedtuple


grammar_text = (r'''
formula = ( _ car_assign _ ) / ( _ available_car _ ) / ( _ pick_up_time _) / ( _ drop_off_time _ ) / ( _ tree_visit _ ) / ( _ capacity _ ) / ( _ congestion _ ) / ( _ exclude _ ) / ( _ reassign _ ) / ( _ multi_pass _ ) / ( _ occupancy _) / ( _ stops_pick_up _) / ( _ stops_drop_off _) / ( _ reward _) / ( _ decomp_reward_1 _) / ( _ decomp_reward_2 _) / ( _ est_arrival _) / ( _ degree_vio_delay _ ) / ( _ degree_vio_adv _ ) / ( _ chance_vio_delay _ ) / ( _ chance_vio_adv _ ) / ( _ cap_vio _ ) / ( _ cap_vio_quant _ ) / ( _ comp_time_vio _ ) / ( _ comp_time_pct _ ) / ( _ comp_reward _ ) / ( _ comp_num_stops _ ) / ( _ additional_search _ )
paren_formula = "(" _ formula _ ")"
available_car = _ "availablecar" _ "(" value ")" _
car_assign = _ "car" _ "(" value ")" _
pick_up_time = _ "tp" _ "(" value ")" _
drop_off_time = _ "td" _ "(" value ")" _
tree_visit = _ "N" _ "(" value "," value ")" _
capacity = _ "C" _ "(" value ")" _
congestion = _ "Cong" _ "(" value ")" _
exclude = _ "exclude" _ "(" value ")" _
reassign = _ "reassign" _ "(" value ")" _
multi_pass = _ "multi" _ "(" value ")" _
additional_search = _ "search" _ "(" value ")" _
occupancy = _ "O" _ "(" value "," value ")" _
stops_pick_up = _ "sp" _ "(" value "," value ")" _
stops_drop_off = _ "sd" _ "(" value "," value ")" _
reward = _ "r" _ "(" value ")" _
decomp_reward_1 = _ "rd1" _ "(" value ")" _
decomp_reward_2 = _ "rd2" _ "(" value ")" _
est_arrival = _ "eta" _ "(" value ")" _
degree_vio_delay = _ "viod" _ "(" time "," est_arrival ")" _
degree_vio_adv = _ "vioa" _ "(" time "," est_arrival ")" _
chance_vio_delay = _ "pctd" _ "(" time "," est_arrival ")" _
chance_vio_adv = _ "pcta" _ "(" time "," est_arrival ")" _
cap_vio = _ "vcv" _ "(" capacity "," occupancy ")" _
cap_vio_quant = _ "vcvq" _ "(" capacity "," occupancy ")" _
comp_time_vio = _ "Phi1" _ "(" vio "," vio ")" _
comp_time_pct = _ "Phi2" _ "(" pct "," pct ")" _
comp_reward = _ "Phi3" _ "(" rwd "," rwd ")" _
comp_num_stops = _ "Phi4" _ "(" stops "," stops ")" _
time = pick_up_time / drop_off_time
vio =  degree_vio_delay / degree_vio_adv
pct =  chance_vio_delay / chance_vio_adv
rwd = reward / decomp_reward_1 / decomp_reward_2
stops =  stops_pick_up / stops_drop_off
value = _ ~r"\d+" _
_ = ~r"\s"*
''')


_grammar = Grammar(grammar_text)


class QueryLogicVisitor(NodeVisitor):

    def visit_formula(self, node, children):
        return children[0][1]

    def visit_paren_formula(self, node, children):
        return children[2]

    def visit_available_car(self, node, children):
        _,_,_,_, request, _,_ = children
        return AvailableCar(int(request))

    def visit_car_assign(self, node, children):
        _,_,_,_, request, _,_ = children
        return CarAssign(int(request))

    def visit_pick_up_time(self, node, children):
        _,_,_,_, request, _,_ = children
        return PickUpTime(int(request))

    def visit_drop_off_time(self, node, children):
        _,_,_,_, request, _,_ = children
        return DropOffTime(request)

    def visit_tree_visit(self, node, children):
        _,_,_,_, tnode, _, vehicle, _,_ = children
        return TreeVisit(int(tnode), int(vehicle))

    def visit_capacity(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return Capacity(int(tnode))

    def visit_congestion(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return Congestion(int(tnode))

    def visit_exclude(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return Exclude(int(tnode))

    def visit_reassign(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return Reassign(int(tnode))

    def visit_multi_pass(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return MultiPass(int(tnode))

    def visit_additional_search(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return AdditionalSearch(int(tnode))

    def visit_occupancy(self, node, children):
        _,_,_,_, tnode, _, vehicle, _,_ = children
        return Occupancy(tnode,int(vehicle))

    def visit_stops_pick_up(self, node, children):
        _,_,_,_, tnode, _, vehicle, _,_ = children
        return StopsPickUp(tnode,vehicle)

    def visit_stops_drop_off(self, node, children):
        _,_,_,_, tnode, _, vehicle, _,_ = children
        return StopsDropOff(tnode,vehicle)

    def visit_reward(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return Reward(tnode)

    def visit_decomp_reward_1(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return DecompReward1(tnode)

    def visit_decomp_reward_2(self, node, children):
        _,_,_,_, tnode, _,_ = children
        return DecompReward2(tnode)

    def visit_est_arrival(self, node, children):
        _,_,_,_, request, _,_ = children
        return ETA(request)

    def visit_degree_vio_delay(self, node, children):
        _,_,_,_, time, _, est, _,_ = children
        return DegreeVioDelay(time, est)

    def visit_degree_vio_adv(self, node, children):
        _,_,_,_, time, _, est, _,_ = children
        return DegreeVioAdv(time, est)

    def visit_chance_vio_delay(self, node, children):
        _,_,_,_, time, _, est, _,_ = children
        return ChanceVioDelay(time, est)

    def visit_chance_vio_adv(self, node, children):
        _,_,_,_, time, _, est, _,_ = children
        return ChanceVioAdv(time, est)

    def visit_comp_time_vio(self, node, children):
        _,_,_,_, left, _, right, _,_ = children
        return CompVioTime(left, right)

    def visit_comp_time_pct(self, node, children):
        _,_,_,_, left, _, right, _,_ = children
        return CompPctTime(left, right)

    def visit_comp_reward(self, node, children):
        _,_,_,_, left, _, right, _,_ = children
        return CompReward(left, right)

    def visit_comp_num_stops(self, node, children):
        _,_,_,_, left, _, right, _,_ = children
        return CompNumStops(left, right)

    def visit_time(self, node, children):
        return children[0]

    def visit_vio(self, node, children):
        return children[0]

    def visit_pct(self, node, children):
        return children[0]

    def visit_rwd(self, node, children):
        return children[0]

    def visit_stops(self, node, children):
        return children[0]

    def visit_cap_vio(self, node, children):
        _,_,_,_, cap, _, occ, _,_ = children
        return CapacityVio(cap, occ)

    def visit_cap_vio_quant(self, node, children):
        _,_,_,_, cap, _, occ, _,_ = children
        return CapacityVioQuant(cap, occ)

    def visit_value(self, node, children):
        return Constant(node.text)

    def generic_visit(self, node, children):
        if children:
            return children



class AvailableCar(namedtuple('car',['request'])):
    def __repr__(self):
        return "availablecar({})".format(self.request)

class CarAssign(namedtuple('car',['request'])):
    def __repr__(self):
        return "car({})".format(self.request)

class PickUpTime(namedtuple('tp',['request'])):
    def __repr__(self):
        return "tp({})".format(self.request)

class DropOffTime(namedtuple('td',['request'])):
    def __repr__(self):
        return "td({})".format(self.request)

class TreeVisit(namedtuple('N',['dep', 'vehicle'])):
    def __repr__(self):
        return "N({},{})".format(self.dep, self.vehicle)

class Capacity(namedtuple("C", ['vehicle'])):
    def __repr__(self):
        return "C({})".format(self.vehicle)

class Congestion(namedtuple("Cong", ["request"])):
    def __repr__(self):
        return "Cong({})".format(self.request)

class Exclude(namedtuple("exclude", ["vehicle"])):
    def __repr__(self):
        return "exclude({})".format(self.vehicle)

class Reassign(namedtuple("reassign", ["vehicle"])):
    def __repr__(self):
        return "reassign({})".format(self.vehicle)

class MultiPass(namedtuple("multi", ["request"])):
    def __repr__(self):
        return "multi({})".format(self.request)

class AdditionalSearch(namedtuple("search", ['vehicle'])):
    def __repr__(self):
        return "search({})".format(self.vehicle)

class Occupancy(namedtuple("O",["node", "vehicle"])):
    def __repr__(self):
        return "O({},{})".format(self.node, self.vehicle)

class StopsPickUp(namedtuple("sp",["request", "vehicle"])):
    def __repr__(self):
        return "sp({},{})".format(self.request, self.vehicle)

class StopsDropOff(namedtuple("sd",["request", "vehicle"])):
    def __repr__(self):
        return "sd({},{})".format(self.request, self.vehicle)

class Reward(namedtuple("r", ['node'])):
    def __repr__(self):
        return "r({})".format(self.node)

class DecompReward1(namedtuple("rd1", ['node'])):
    def __repr__(self):
        return "rd1({})".format(self.node)

class DecompReward2(namedtuple("rd2", ['node'])):
    def __repr__(self):
        return "rd2({})".format(self.node)

class ETA(namedtuple("eta", ['request'])):
    def __repr__(self):
        return "eta({})".format(self.request)

class DegreeVioDelay(namedtuple("viod", ['time', 'est_arrival'])):
    def __repr__(self):
        return "viod({}, {})".format(self.time, self.est_arrival)
    def children(self):
        return [self.time, self.est_arrival]

class DegreeVioAdv(namedtuple("vioa", ['time', 'est_arrival'])):
    def __repr__(self):
        return "vioa({}, {})".format(self.time, self.est_arrival)
    def children(self):
        return [self.time, self.est_arrival]

class ChanceVioDelay(namedtuple("pctd", ['time', 'est_arrival'])):
    def __repr__(self):
        return "pctd({}, {})".format(self.time, self.est_arrival)
    def children(self):
        return [self.time, self.est_arrival]

class ChanceVioAdv(namedtuple("pcta", ['time', 'est_arrival'])):
    def __repr__(self):
        return "pcta({}, {})".format(self.time, self.est_arrival)
    def children(self):
        return [self.time, self.est_arrival]

class CapacityVio(namedtuple("vcv", ['cap', 'occ'])):
    def __repr__(self):
        return "vcv({}, {})".format(self.cap, self.occ)
    def children(self):
        return [self.cap, self.occ]

class CapacityVioQuant(namedtuple("vcvq", ['cap', 'occ'])):
    def __repr__(self):
        return "vcvq({}, {})".format(self.cap, self.occ)
    def children(self):
        return [self.cap, self.occ]

class CompVioTime(namedtuple("Phi1", ['left', 'right'])):
    def __repr__(self):
        return "Phi1({}, {})".format(self.left, self.right)
    def children(self):
        return [self.left, self.right]

class CompPctTime(namedtuple("Phi2", ['left', 'right'])):
    def __repr__(self):
        return "Phi2({}, {})".format(self.left, self.right)
    def children(self):
        return [self.left, self.right]

class CompReward(namedtuple("Phi3", ['left', 'right'])):
    def __repr__(self):
        return "Phi3({}, {})".format(self.left, self.right)
    def children(self):
        return [self.left, self.right]

class CompNumStops(namedtuple("Phi4", ['left', 'right'])):
    def __repr__(self):
        return "Phi4({}, {})".format(self.left, self.right)
    def children(self):
        return [self.left, self.right]

class Constant(float):
    pass



def parse(input_str):
    return QueryLogicVisitor().visit(_grammar.parse(input_str))

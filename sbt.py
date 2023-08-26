import ast
import json

source_code_demo = """
def runCommand(mergeErrorIntoOutput, commands):
    myString = "Hello"
    for word in myString:
        print(word)
"""

# 打印解析得到的AST

def nodeBranch(field,sbt,name):
    if type(field) in (int,float,str): 
        sbt = sbt+["(",str(name)+'_'+str(field),")",str(field)]
    else:
        sbt = sbt+["(",name]
        sbt = SBT(field,sbt)
        sbt = sbt+[")",name]
    return sbt


def SBT (node,sbt):
    sbt = sbt+["(",node.__class__.__name__]
    for name, field in ast.iter_fields(node):
        if field :
            if type(field)==list:
                for item in field:
                    sbt = nodeBranch(item,sbt,name)
            else: sbt = nodeBranch(field,sbt,name)
    sbt = sbt+[")",node.__class__.__name__]
    return sbt

#sbt = SBT(parsed_ast,[])
#print(sbt)

def nodeBranch2(field,sbt,name,ids,idss):
    if type(field) in (int,float,str): 
        sbt = sbt+[str(name)+'_'+str(field),str(field)]
        idss = idss+[ids,ids]
    else:
        sbt = sbt+[name]
        idss = idss+[ids]
        sbt,idss = SBT2(field,sbt,ids+1,idss)
        sbt = sbt+[name]
        idss = idss+[ids]
    return sbt,idss


def SBT2 (node,sbt,ids,idss):
    sbt = sbt+[node.__class__.__name__]
    idss = idss+[ids]
    if hasattr(node, "_fields"):
        for name, field in ast.iter_fields(node):
            if field :
                if type(field)==list:
                    for item in field:
                        sbt,idss = nodeBranch2(item,sbt,name,ids+1,idss)
                else: sbt,idss = nodeBranch2(field,sbt,name,ids+1,idss)
    sbt = sbt+[node.__class__.__name__]
    idss = idss+[ids]
    return sbt,idss



#sbt,idss = SBT2(parsed_ast,[],0,[])
#print(sbt)
#print(idss)



'''
Module(
    body=[
        FunctionDef(name='greet', 
                    args=arguments(posonlyargs=[], args=[arg(arg='name')], kwonlyargs=[], kw_defaults=[], defaults=[]), 
                    body=[Expr(value=Call(func=Name(id='print', ctx=Load()), 
                                          args=[JoinedStr(values=[Constant(value='Hello, '), 
                                                                  FormattedValue(value=Name(id='name', ctx=Load()), conversion=-1), 
                                                                  Constant(value='!')])], keywords=[]))], 
                    decorator_list=[])
        , Expr(value=Call(func=Name(id='greet', ctx=Load()), args=[Constant(value='Alice')], keywords=[]))
        ], type_ignores=[]
)
'''
def isNode (node):
    if type(node) == str or type(node) == int: return False
    if type(node) in [ast.Load,ast.Store]:
        return False
    else: return True

def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def getConstant(node):
    if type(node) == ast.Constant: 
        if isinstance(node.value,int):
            return 'NUM_'
        elif isinstance(node.value,bool):
            return 'BOOL_'
        elif isinstance(node.value,str):
            return 'STR_'
    if type(node) == ast.Name: return node.id
    if type(node) == ast.arg: return node.arg
    else: return False


def process_source(node,nodeList,name):
    childArr = []
    # When traversing a list
    if type(node)==list:
        for item in node:
            nodeList,childId = process_source(item,nodeList,name)
            childArr.append(childId)
            childArr = flatten_list(childArr)
        return nodeList,childArr
    id = -1
    # When traversing to a node
    if isNode(node):
        id = 0 if len(nodeList)==0 else  nodeList[-1]['id']+1
        if node.__class__.__name__ == 'Name':
            nodeList.append({'id':id , 'type':name})
        else: nodeList.append({'id':id , 'type':node.__class__.__name__})
        # Get the contents of a node
        value = getConstant(node)
        if value: nodeList[id]['value'] = value
    # Traverse the child nodes
    if hasattr(node, "_fields"):
        for mname, field in ast.iter_fields(node):
            if field :
                nodeList,childId = process_source(field,nodeList,mname)
                childArr.append(childId)
                childArr = flatten_list(childArr)
    if isNode(node) and len(childArr)>0:
        nodeList[id]['children'] = childArr
    if id == -1:
        return nodeList,childArr
    else:
        return nodeList,id

def SBT_1(cur_root_id, node_list):
    cur_root = node_list[cur_root_id]
    tmp_list = []
    tmp_list.append("(")
    str = cur_root['type']
    tmp_list.append(str)
    if 'children' in cur_root:
        chs = cur_root['children']
        for ch in chs:
            tmp_list.extend(SBT_1(ch, node_list))
    tmp_list.append(")")
    tmp_list.append(str)
    return tmp_list

def SBT_2(cur_root_id, node_list, zindex, idx):
    cur_root = node_list[cur_root_id]
    tmp_list = []
    idx.append(zindex)
    str = cur_root['type']
    tmp_list.append(str)
    zindex_child = zindex
    if 'children' in cur_root:
        chs = cur_root['children']
        for ch in chs:
            tem_list_new,idx, zindex_child = SBT_2(ch, node_list,zindex_child+1,idx)
            tmp_list.extend(tem_list_new)
    idx.append(zindex)
    if 'value' in cur_root:
        tmp_list.append(cur_root['value'])
    else:
        tmp_list.append(str)
    return tmp_list, idx, zindex_child


def start (type,source_code):
    parsed_ast = ast.parse(source_code)
    #print(ast.dump(parsed_ast, indent=4))
    if type == 1:
        sbt = SBT(parsed_ast,[])
        return sbt
    if type == 2:
        sbt,idss = SBT2(parsed_ast,[],0,[])
        return sbt,idss
    if type == 3:
        out = process_source(parsed_ast,[],'')[0]
        #for item in out:
        #    print(item)
        sbt,idss, zindex = SBT_2(0,out,0,[])
        return sbt,idss

print(start(3,source_code_demo))

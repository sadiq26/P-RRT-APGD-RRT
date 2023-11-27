obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]

start_pos=(0,0)
end_pos=(50,50)



def get_centroid(obs):
    '''Find the center of obstacle list 
     \n input: Obstacle list in format [[ox,oy,w,h],[]]
     \n returns: obs centroid [[],[]]'''
    obs_centroid=[]
    for (ox, oy, w, h) in obs:
        centroid_list=[[ox+w/2,oy+h/2]]
        obs_centroid.append(centroid_list)
    return obs_centroid




def get_obs_vertex(obs_rec):
    '''Gets obstacle vertice + clearance'''
    delta = 0   #Clearance value
    obs_list = []

    for (ox, oy, w, h) in obs_rec:
        vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
        obs_list.append(vertex_list)

    return obs_list

def vertex_2_point(obs_vertix):
    vertix_point=[]
    for v in obs_vertix:
        for (x_p,y_p) in v:
            vertix_point.append([x_p,y_p])
    return vertix_point


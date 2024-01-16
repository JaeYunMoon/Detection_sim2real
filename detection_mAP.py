import os,glob
import cv2 
from collections import defaultdict, Counter


class mAP():
    def __init__(self,gt_paths:list,predict_paths:list,class_dict:dict) -> None:
        """
        gt_path = 정답.txt 
        gt = class xmin ymin xmax ymax 

        predict_paths = 예측값.txt 
        predict = class confidence xmin ymin xmax ymax 

        모든 txt들은 xmin,ymin,xmax,ymax 박스로 구성 되어 있어야 함

        """
        assert len(gt_paths) != 0,"gt_paths empty" 
        self.gtpaths = gt_paths
        self.prpaths = predict_paths
        self.cd = class_dict

    def iou(self,box1, box2):
        # Calculate the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate the area of intersection
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate the areas of the bounding boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate the IOU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou
    def mAP_processing(self,gt_pths,prd_path):
        ap = defaultdict(list)
        conf = defaultdict(list)
        total = []

        for gpth in gt_pths:
            gname = os.path.basename(gpth)
            pname = prd_path + gname 
            # print(gpth)

            gt_txt = open(gpth,"r")
            prd_txt = open(pname,"r")

            gt = gt_txt.readlines()
            prd = prd_txt.readlines()

            if len(prd):
                for g in gt:
                    c,x,y,xm,ym = g.split(" ")
                    # print(g)
                    c = int(c)
                    xyxy = [int(x) for x in [x,y,xm,ym.replace("\n","")]]
                    # print(c,xyxy)
                    total.append(c)
                    for pr in prd:
                        pc,cf,px,py,pxm,pym = pr.split(" ")
                        pc = int(pc)
                        pxyxy = [int(y) for y in [px,py,pxm,pym.replace("\n","")]]
                        
                        
                        iu = self.iou(pxyxy,xyxy)
                        if pc == c :
                            
                            ap[pc].append(1)
                            conf[pc].append(cf)

                        elif pc!=c and iu != 0:
                            ap[pc].append(0)
                            conf[pc].append(cf)
            else:
                #print("s")
                for g in gt:
                    c,x,y,xm,ym = g.split(" ")
                    c = int(c)
                    total.append(c)
        return ap,conf,total 
    
    def sort_lists_based_on_other(self,input_list, reference_list):
        # confidence 값으로 predict값 정렬하는 함수 
        sorted_reference = sorted(reference_list, reverse=True)
        sorted_input = [x for _, x in sorted(zip(reference_list, input_list), reverse=True)]
        return sorted_input, sorted_reference
    
    def calculate_average_precision(self,precisions, recalls):
        # ap 계산하는 함수 
        # Precision-Recall 곡선 아래 영역 계산 (AP 계산)
        ap = 0
        
        for i in range(1, len(precisions)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]

        if len(recalls) == 1:
            ap = recalls[0] * precisions[0]
        return ap
    
    def caculate(self):
        prediction,conf,gt = self.mAP_processing(self.gtpaths,self.prpaths)
        ap = {} 
        gt_count = Counter(gt)
        

        for i in sorted(gt_count.keys()):
            recalls = [] 
            precisions = [] 
            prc = 0 
            co = 0 
            if not prediction[i]:
                ap[self.cd[i]] = 0 

            sortprd,sortconf = self.sort_lists_based_on_other(prediction[i],conf[i])
            
            for j in sortprd:
                prc += 1 
                co += j 
                
                recalls.append(co/gt_count[i])
                precisions.append(co/prc)
            ex_ap = self.calculate_average_precision(precisions,recalls)
            
        
            ap[self.cd[i]] = round(ex_ap,6)    
        all_sum = 0
        for k in ap.keys():
            all_sum += ap[k]
        map = all_sum / len(ap.keys())

        return ap, round(map,4)

def makeDetectResult(pths,yolo_model,save_path,label_dict):
    
    img_save = path_confirm(save_path+"/img_result/")
    predic_save = path_confirm(save_path+"/result/")
    gt_save = path_confirm(save_path+"/gt/")
    font =  cv2.FONT_HERSHEY_PLAIN
    red = (0,0,255)
    green = (0,255,0)
    blue = (255,0,0)
    yellow = (255,255,0)
    gt_clss = [] 
    gt_xyxy = [] 
    for i in range(len(pths)):
        sub_cl = [] 
        sub_xyxy = [] 

        gt = pths[i].replace("images","labels").replace(".jpg",".txt")

        pn = os.path.basename(pths[i])
        img = pths[i]
        im = cv2.imread(img,cv2.COLOR_BGR2RGB)
        y,x,ch = im.shape 
        results = yolo_model(im)
        f = open(gt,"r")
        lines = f.readlines()
        clss = []
        confs = [] 
        xyxy = []  
        
        for line in lines:
            ln = line.split(" ")
            c,xc,yc,w,h = ln 
            xc,yc,w,h = float(xc)*x,float(yc)*y,float(w)*x,float(h)*y
            sub_cl.append(int(c))
            xmin = int((xc - (w//2)))
            xmax = int((xc + (w//2)))
            ymin = int((yc - (h//2)))
            ymax = int((yc + (h//2)))
            sub_xyxy.append([xmin,ymin,xmax,ymax])
        gt_clss.append(sub_cl)
        gt_xyxy.append(sub_xyxy)

        for result in results:
            bx = result.boxes
            
            clss.append(bx.cls.tolist())
            confs.append(bx.conf.tolist())
            xyxy.append(bx.xyxy.tolist())
        f = open(predic_save + pn.replace(".jpg",".txt"),"w")
        fl = open(gt_save +  pn.replace(".jpg",".txt"),"w")
        for idx in range(len(sub_cl)):
            gt_coor = sub_xyxy[idx]
            im = cv2.rectangle(im,(gt_coor[0],gt_coor[1]),(gt_coor[2],gt_coor[3]),green,1)
            
            im = cv2.putText(im,label_dict[sub_cl[idx]],(int(gt_coor[0]-5),int(gt_coor[1]-5)),font,2,green,1,cv2.LINE_AA)

            fl.write(str(sub_cl[idx])+ " ")
            fl.write(" ".join([str(coordi) for coordi in gt_coor])+"\n")

            try:
                prex,prey,prexm, preym = xyxy[0][idx]
                prex,prey,prexm, preym = int(prex),int(prey),int(prexm),int(preym)
                im = cv2.rectangle(im,(prex,prey),(prexm,preym),red,2)
                im = cv2.putText(im,label_dict[int(clss[0][idx])],((prexm+5),int(preym+5)),font,2,blue,1,cv2.LINE_AA)
                print(xyxy[0][idx])
                x = [prex,prey,prexm, preym]
                predic_box = [str(y) for y in x]
                print(x)
                print(predic_box)
                f.write(f"{int(clss[0][idx])}"+ " ")
                f.write(f"{round(float(confs[0][idx]),4)} ")
                f.write(" ".join(predic_box)+ "\n")
                

                

                
            except:
                pass 
            
        cv2.imwrite(img_save+f'{pn}.png',im)
        
        gt_list = glob.glob(gt_save+"*")
    return gt_list, predic_save

def path_confirm(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path 

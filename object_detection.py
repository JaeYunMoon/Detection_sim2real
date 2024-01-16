import argparse
import os 
from util.BoundinBoxes_list_dict import BoundingBoxes
from util.BoundingBox import BoundingBox
from util.Evaluator import Evaluator
from util.general import check_dir,check_save_dir,ValidateFormats,BBType

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt","--gt_folder",dest="gtdir",type=str,default="./gt")
    parser.add_argument("-det","--detection_folder",dest ="detdir",type=str,default="./gt")
    parser.add_argument("-t","--threshold",dest="iou_threshold",type=float,default=0.5)
    parser.add_argument("-sp","--save_path",dest="savePath",default="./result/")
    parser.add_argument("--noplot",dest="showPlat",action="store_false")
    parser.add_argument("--gtformat",dest="gtFormat",default="xyrb")
    parser.add_argument("--detformat",dest="detFormat",default="xyrb")
    # gt,detection fromat 추가 안함.. 
    # 따라서 imgsz size도 추가 안함 
    # 절대 상대 좌표 코드도 작성 안함
    args = parser.parse_args()
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    _gt = check_dir(opt.gtdir,'--gt_folder')
    _det = check_dir(opt.detdir,'--detection_folder')
    assert len(_gt) == len(_det),f"The number of correct txt files do not match. GT,Detect {len(_gt),len(_det)}"
    
    save_path = check_save_dir(opt.savePath)

    gtFormat = ValidateFormats(opt.gtFormat,'--gtformat')
    detFormat = ValidateFormats(opt.detFormat,'--detformat')
    print(gtFormat)
    allBoundingBoxes,allClasses = getBoundingBoxes(files = _gt,bbFormat = gtFormat,isGT=True)
    allBoundingBoxes,allClasses = getBoundingBoxes(files = _det,bbFormat = detFormat,isGT=False,allBoundingBoxes=allBoundingBoxes)
    allClasses.sort()
    # allBoundingBoxes.getBoundingBoxes()
    # [imageName, class, confidence, (bb coordinates XYX2Y2)]
    # for bb in allBoundingBoxes.getBoundingBoxes():
    #     print(bb)
    evaluator = Evaluator() 
    acc_AP = 0
    validClasses = 0
    
    detections = evaluator.PlotPrecisionRecallCurve(allBoundingBoxes,
                                                    IOUThreshold=opt.iou_threshold,
                                                    showAP=True,
                                                    showInterpolatedPrecision = False,
                                                    savePath = save_path,
                                                    showGraphic = opt.showPlat)


    f = open(os.path.join(save_path, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)

def getBoundingBoxes(files,isGT,
                     bbFormat,
                     allBoundingBoxes = None,
                     allClasses = None,
                     imgSize = (0,0)
                     ):
    """
    상대좌표 배제하고 절대좌표만,
    bboxFormat은 xywh 배제, min(xy),max(xy)
    
    """
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = [] 
    for fl in files:
        # print(fl)
        names = os.path.basename(fl).replace(".txt","")
        fh = open(fl,"r")
        fh1 = fh.readlines()
        for line in fh1:
            # print(line)
            if line.replace(' ', '') == '':
                continue
            c,*xyxy = line.split(" ")
            xyxy = [float(x.rstrip()) for x in xyxy]
            # print(xyxy)
            if isGT:
                idClass = (c)
                x = float(xyxy[0])
                y = float(xyxy[1])
                xmax = float(xyxy[2])
                ymax = float(xyxy[3])
                bb = BoundingBox(names,
                                 idClass,
                                 x,
                                 y,
                                 xmax,
                                 ymax,
                                 BBType.GroundTruth,
                                 format = bbFormat
                                 )
            else:
                idClass = (c)
                confidence = float(xyxy[0])
                x = float(xyxy[1])
                y = float(xyxy[2])
                xmax = float(xyxy[3])
                ymax = float(xyxy[4])
                bb = BoundingBox(names,
                                 idClass,
                                 x,
                                 y,
                                 xmax,
                                 ymax,
                                 BBType.Detected,
                                 confidence,
                                 format= bbFormat
                                 )
     
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh.close()
    return allBoundingBoxes, allClasses

                
                

    

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

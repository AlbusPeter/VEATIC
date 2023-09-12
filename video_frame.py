import cv2
import argparse
import os

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='./data/videos/', help='Path to all videos  ')
parser.add_argument('--path_save_video', type=str, default='./data/frames/', help='Path to save the result frames')

args = parser.parse_args()

def save_video_frame():
    path_all_videos = os.listdir(args.path_video)

    if not os.path.exists(args.path_save_video):
        os.makedirs(args.path_save_video)
        
                
    for i, v_path in enumerate(path_all_videos):
        print('{}/{}'.format(i + 1, len(path_all_videos)))
    
        name_video = os.path.basename(v_path)
        
        path_video = os.path.join(args.path_video, v_path)
        path_save_video = os.path.join(args.path_save_video, name_video[:-4])
        
        
        if not os.path.exists(path_save_video):
            os.makedirs(path_save_video)
        
        video_cap = cv2.VideoCapture(path_video)

        
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_fNUMS = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        

        print("cha_fps:{}".format(video_fps))
        print("cha_size:{}".format(video_size))
        print("cha_fNUMS:{}".format(video_fNUMS))
        
        
        i = 0
        video_success, video_frame = video_cap.read()
        
        while video_success:
            cv2.imwrite(path_save_video + "/{}.jpg".format(i), video_frame)
            video_success, video_frame = video_cap.read()
            i += 1
        video_cap.release()


        
if __name__ == "__main__":
    save_video_frame()











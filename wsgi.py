import threading
from app.vis import vis


def visualisation():
    vis.run()
    
if __name__ == "__main__":
    threads = list()
    vis_thread= threading.Thread(target=visualisation)
    threads.append(t)
    print('Starting Thread {}'.format(i))
    vis_thread.start()
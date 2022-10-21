import threading
from app.vis import app
#from app.vis_copy import app2

def visualisation():
    app.run()

def test():
    print ("test______________________________________test____________")
    #app2.run()

if __name__ == "__main__":
    threads = list()
    vis_thread= threading.Thread(target=visualisation)
    test_thread= threading.Thread(target=test)

    threads.append(vis_thread)
    print('Starting Thread {}'.format(i))
    vis_thread.start()

    threads.append(test_thread)
    print('Starting Thread {}'.format(i))
    test_thread.start()
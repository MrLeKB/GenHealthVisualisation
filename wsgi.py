import threading
from app.vis import app
from app.vis_copy import app2

def visualisation():
    app.run()

def test():
    print ("test______________________________________test____________")
    app2.run()

if __name__ == "__main__":
    threads = list()
    vis_thread= threading.Thread(target=visualisation)
    threads.append(t)
    print('Starting Thread {}'.format(i))
    vis_thread.start()

    vis_thread2= threading.Thread(target=test)

    threads.append(t)
    print('Starting Thread {}'.format(i))
    vis_thread2.start()
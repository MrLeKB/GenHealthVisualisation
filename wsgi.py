import threading
from app.vis import app
#from app.vis_copy import app2

def visualisation():
    app.run()

def test():
    print ("test______________________________________test____________")
    #app2.run()

if __name__ == "__main__":
    test()
    visualisation()
    test()
    # threads = list()
    # vis_thread= threading.Thread(target=visualisation)
    # test_thread= threading.Thread(target=test)

    # print('Starting Thread 1')
    # vis_thread.start()
    # print('Starting Thread 2')
    # test_thread.start()

    # vis_thread.join()
    # test_thread.join()




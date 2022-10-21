# import threading
# from app.vis import app
# #from app.vis_copy import app2

# def visualisation():
#     app.run()

# def test(sad):
#     print ("test______________________________________test____________")
#     #app2.run()

# if __name__ == "__main__":
#     threads = list()
#     vis_thread= threading.Thread(target=visualisation)
#     test_thread= threading.Thread(target=test)

#     print('Starting Thread 1')
#     vis_thread.start()
#     print('Starting Thread 2')
#     test_thread.start()

#     vis_thread.join()
#     test_thread.join()


import random
import threading
import time

def sleeper():
    time.sleep(random.randrange(1, 20))

if __name__ == '__main__':
    # create and start our threads
    threads = list()
    for i in range(4):
        t = threading.Thread(target=sleeper) # pass in the callable
        threads.append(t)
        print('Starting Thread {}'.format(i))
        t.start()

    # wait for each to finish (join)
    for i, t in enumerate(threads):
        t.join()
        print('Thread {} Stopped'.format(i))


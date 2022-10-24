from app.vis import app
# #from app.vis_copy import app2

def visualisation():
    print("first thread______________________________________________")
    #app.run()

def test():
      return "<h1>Welcome to CodingX</h1>"


if __name__ == "__main__":
    test
    #visualisation()
    # test()
    # visualisation()
    # test()
    # threads = list()
    # vis_thread= threading.Thread(target=visualisation)
    # test_thread= threading.Thread(target=test)

    # print('Starting Thread 1')
    # vis_thread.start()
    # print('Starting Thread 2')
    # test_thread.start()

    # vis_thread.join()
    # test_thread.join()




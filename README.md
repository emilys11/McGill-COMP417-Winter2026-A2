# McGill-COMP417-Winter2026-A2
The starter code for your Assignment 2 coding submission.

# Getting Started

You will likely need to install some Python library dependencies. We recommend starting from a somewhat recent and clean Python environment. We have tested that things work on:
- Python 3.13.11 on Ubuntu via latest miniconda
- Python 3.13.11 on Mac via latest miniconda
- Python 3.8.10 native install on Ubunutu

We found some incompatible/missing features in older versions:
- Python 3.8.5 native install on Ubuntu (needed to run sudo apt update, sudo apt upgrade to get native Python to 3.8.10 level, reinstall/upgrade libraries)

## First run of the code

Launch with:
$ python arm_rrt.py

You should see something like this GUI rendering:
<img width="300" alt="image" src="https://github.com/user-attachments/assets/719cdeb7-3460-4c8b-9d07-e88061b590f3" />

This represents a "stub" RRT planner running on the arm. Instead of building a proper tree, it just connects nodes randomly to the start and goal poses. However, it uses all the same supporting code as our working assignment solution, so it will give you a good start. Whenever you see the GUI appearing and showing this image, you are ready to start coding. If you have any errors, missing libraries, etc, try to fix them or post on Ed or office hours if you need help.

## Read and Familiarize yourself with the files

There is one main file, arm_rrt.py, which you will work in completely. You should not modify the other files, which contain the Python interface to our robot and simulator, in robots.py, and some resources for the robot's geometry, in the folder gen3lite_urdf. You can browse those, but should not really need to open them if all goes well.

Within arm_rrt.py, your primary job is to delete/fix the code in the plan() function, and replace this with a working RRT. It can be a single tree version or RRT-Connect (recommended), such that you are able to plan through the three provided environments, selected with the --environment= option:
- Free (no obstacles)
- Easiest (one tier of balloons around the goal)
- Hardest (two tiers, blocking goal and start)

Look at the overall code structure, Node and Tree helper classes, constructor code flow, helper functions within the RRT class and the main() at the bottom of the file. When you understand what you're working with, it's time to give your solution a try!

## Debugging Tips and Tricks

- You can move the camera. Buttons depend on your OS and availability of mouse/track-pad/whatever. For me, mouse wheel scroll is zoom, pressing ctrl and mouse dragging is rotate, and holding the middle mouse button and dragging is translate.
- Do increase the max_iter variable to get especially the Hardest environment solved. My solution required between 3 and 8 thousand nodes to have a solution. Changing this value and looking at the run-times lets you answer one of the written questions.
   - Note that, when trying to debug the basics of your tree building, it may be helpful to use smaller values that don't cause a full plan to be found, but do allow you to see how the first few nodes are added on. For example, is the CONNECT phase of your RRT-CONNECT code really going in the right direction?

## Solve Planning for the Three Provided Environments

When you're code succeeds, you should be able to clearly see (a) planning tree(s) created, and the path from start to goal highlighted in yellow. For example, our solution to Hardest looks as follows:

<img width="300" alt="image" src="https://github.com/user-attachments/assets/b30592b4-88c1-4668-a4df-e53eae9a9467" />

## Document Your Learning Experience

The A2 PDF document has some questions that you need to answer from your own knowledg and by running your solution code. Prepare a PDF response capturing some screenshots of your planner in action and with your text responses. Then hand in on My Courses. Don't forget to check that needed modifications are only within the arm_rrt.py file. If you temporarily created other files or modified robots.py, try a fresh clone of this repo and make sure your planner still works well there.



 


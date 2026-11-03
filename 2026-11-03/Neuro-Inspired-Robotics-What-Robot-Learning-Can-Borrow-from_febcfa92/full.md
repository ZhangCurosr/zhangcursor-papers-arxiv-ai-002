# Neuro Inspired Robotics: What Robot Learning Can Borrow from the Sensorimotor Brain

Ruihan Zhang<sup>∗</sup> Davy Deng<sup>†1</sup>

<sup>1</sup>MIT

## Abstract

Robot policies still fail at everyday manipulation: an unfamiliar task, an object never held before, a scene that changes mid-motion, a skill that should improve with use. They fail despite pretraining on over a million trajectories, pooled from dozens of robots at great expense, while a child acquires all of these abilities within a few years of learning from a single body. This survey argues that the gap is one of design, and proposes designs for the next generation of robot models, drawn from how the nervous system solves the same problems. From the body we take compliance: the policy commands stifness alongside position, so the controller yields where the task allows and presses where it must, which greatly lowers the demand for precise position output. From the visual system, where a fast pathway for action runs in parallel with the slow pathway for recognition, we take a dedicated fast vision system for motor behavior: it precomputes the quantities the policy needs and drives a reactive policy at camera rate. From the cerebellum we take a predictor beside the policy that forecasts the consequence of each command and recalibrates from its own error. From motor cortex we take a single model that composes action units from primitives, verifies each at a sensory boundary, practices the weakest unit with a hypothesis in mind, and reasons each attempt closer to the goal. Proven primitives are then consolidated from code into weights, so they run fluently and respond quickly the next time. Together these describe a robot that acts on its own initiative, the basis of deployment and recursive self-improvement, and we ofer each proposal as a hypothesis to be tested.

Keywords: robot learning; embodied AI; neuroscience; biological priors; vision-language-action models; world models; compliance control; tactile sensing; motor primitives; recursive self-improvement

## Contents

## 1 Introduction

2 Action 6   
2.1 Robotics 6   
2.1.1 The policy emits a position; the servo stack supplies the force 6   
2.1.2 Compliance can be added beneath the policy, in hardware, torque, or target 7   
2.1.3 Action chunking decouples control rate from inference rate, at the cost of blindness mid-chunk   
and seams between chunks 7   
2.1.4 Token or number, absolute or delta, the representation changes how the position is encoded 8   
2.2 Biology 8   
2.2.1 The motor neuron integrates cortical, brainstem, and reflex drive into one command . 8   
2.2.2 The population firing rate of a muscle’s motor neurons is translated into efort 8   
2.2.3 Opposing muscle pairs set torque and stifness separately 8   
2.2.4 Slip during a grasp is caught by a reflex that corrects grip force, with no position in the loop 9   
2.3 Inspiration 9   
2.3.1 Use compliance so the world supplies what the policy would otherwise compute 9   
2.3.2 On the position-controlled arms most policies run on, keep the position output, add a stifness   
beneath it, and let touch set where to yield and how hard to press 10   
3 Touch 11   
3.1 Robotics 11   
3.1.1 No sensor family matches human skin on resolution, speed, and force at once 11   
3.1.2 Policies now take touch as an input, but data and control rate hold it back 12   
3.2 Biology 12   
3.2.1 Four receptor classes encode touch in spike timing 12   
3.2.2 Successive contacts build an object representation beyond geometry 14   
3.2.3 Tactile reflexes correct grip within about 75 ms, faster than any visual correction 15   
3.2.4 The brain predicts touch and acts on the discrepancy 16   
3.2.5 Descending control suppresses predictable and irrelevant tactile channels 16   
3.3 Inspiration 16   
3.3.1 Represent touch as a force field first, and let it drive compliance control 17   
3.3.2 Train the model to predict the tactile consequence of each command 17   
3.3.3 Probe actively to test hypotheses about properties that cannot be seen 17   
Vision 19   
4.1 Robotics 19   
4.1.1 The pretraining objective decides what the tokens keep 19   
4.1.2 The VLM or video world-model backbone acts as an extended vision encoder 20   
4.1.3 Current vision stacks are slow, blind to velocity, and miss fine detail 20   
4.2 Biology 20   
4.2.1 The retina compresses before transmitting, and recognition and action get separate streams 20   
4.2.2 The dorsal stream is modeled as a bank of parallel channels, one per class of action 21   
4.2.3 Vision selects one object, holds it on the fovea, and tracks its motion 22   
4.2.4 Each dorsal channel re-expresses visual information toward the frame its efector needs 23   
4.2.5 Vision is slow, so the brain acts on a prediction and corrects it later 24   
4.3 Inspiration 24   
4.3.1 Add a fast streaming encoder for control beside the slow one for recognition 24   
4.3.2 Read the numbers and the decisions an action unit needs of one encoded frame 25   
4.3.3 Move the camera to see better, and train across viewpoints 25   
5 Is the cerebellum the brain’s world model? 27   
5.1 Robotics 27   
5.1.1 World Action Models train the policy under a world-prediction loss 27   
5.1.2 Test-time adaptation stores recent experience in a latent, in temporary weights, or in the context 28   
5.2 Biology 29   
5.2.1 A three-synapse, very wide circuit predicts before the senses report 30   
5.2.2 Error signals recalibrate the cerebellum locally and fast 30   
5.2.3 The prediction amends the action in flight or returns to cortex 31   
5.3 Inspiration 32   
5.3.1 Consult the predictor as a tool when it is slo 32   
5.3.2 Run the predictor in the loop when it is fast 32   
5.3.3 Make the predictor shallow, wide, and sparse, so it is fast and adapts at test time 32   
5.3.4 Tag every joint and sensor with a body-part embedding so the map transfers 33   
Planning and command: how the brain divides what a VLA does in one network 34   
6.1 Robotics 34   
6.1.1 A VLA is a backbone plus action head 34   
6.1.2 Three training stages shape the VLA, and none writes deployment experience back 35   
6.1.3 A planner decomposes the task into subgoals for a fast executor 36   
6.1.4 A multimodal large language model writes the policy as code 36   
6.1.5 Experience enters the model in context, as skills, or in the weights, and no loop yet runs without   
36   
6.2 Biology 37   
6.2.1 Prefrontal cortex sets the goal, SMA holds the action units, and ACC keeps the place 37   
6.2.2 Motor cortex holds sensor-tuned primitives, and a competition among premotor candidates   
chooses which to compose 37   
6.2.3 The cerebellum’s forecast amends the current unit and times the next 38   
6.2.4 Practice turns the composition into a habit, selected by dopamine and consolidated ofline 39   
6.3 Inspiration 39   
6.3.1 Let the model compose action units from reusable primitives and verify each at its boundary 40   
6.3.2 Treat each primitive as a parameterized template 40   
6.3.3 Let the model run each unit by Cartesian control or by a reactive policy head 40   
6.3.4 Add a consolidation agent that graduates proven primitives from code into weights 41   
6.3.5 Design the next architecture so that primitives are experts routed by the plan and share embeddings 41   
6.3.6 Add a practice agent that rehearses weak skills and tests what consolidation wrote 42   
6.3.7 Mastering a new skill rests on judging which direction moves it closer to success 43   
Conclusion 44

## 1 Introduction

A seven-year-old opens drawers, catches a flying toy, and plays the piano, and acquired every one of these skills from a single body over a single childhood. In contrast, robot policies trained on over a million trajectories pooled across dozens of robots still fail at tasks this simple (Figure 1) [296, 42, 210]. Scale alone cannot explain the gap, because the child never saw more than seven years of data. The diference is in how the nervous system is organized, which makes the child more dexterous, more intelligent, and able to keep improving. This survey asks how that sensorimotor skill develops and which parts of the design robotics can borrow. It is built around three questions:

![](images/7a664f01b1a1c9a3e611e1cb6bc8236b7bc3e896cc0e274f3bf146c21c0683b3.jpg)  
Figure 1: A child has three abilities that current robot policies lack. Top: a boy opens an unfamiliar drawer on the first try, while a policy finds no training handle to copy. Middle: a girl catches a flying toy, while a robot is still reasoning about shape and coordinates when the toy lands. Bottom: the boy’s piano playing improves with days of practice, while a robot never trained on the piano cannot learn it or improve over time, because its learning stopped the day it was deployed.

How does a human solve a task it has never seen? Never from scratch. A policy trained end to end has nothing to copy when the scene falls outside its training data (Figure 1). The brain instead uses reasoning to split the task into a sequence of action units and composes each from a small repertoire of primitives such as reach, orient, and grasp, each a verifiable goal and an action plan, so an unfamiliar drawer becomes reach, grasp, and pull on the first try. Each unit ends at a sensory event, the hand closing on the handle, and the brain checks for it before releasing the next. When the check fails, it reasons to the cause and revises the plan (Section 6).

Why does a human act so fast? Robotics debates how to pair a large reasoning model, powerful but slow, with a small action policy, fast but shallow, so the system is quick, capable, and still fits on the GPU a robot can carry. The brain escapes the trade-of by keeping deliberation of the path from sensing to action. A second visual stream, largely separate from the one that recognizes objects, guides the next movement (Section 4). Its commands can be coarse because the body is compliant: the controller only decides where motion must be exact and where external forces can carry it (Section 2). Touch corrects a slipping grip by reflex within a hundred milliseconds, before vision registers the slip (Section 3). And on the leading account of the cerebellum, a predictor forecasts each command’s consequence before the sensors report it, so the response is ready when the world catches up (Section 5).

How does a human keep self-improving? A robot policy is trained once before deployment, and naive fine-tuning risks erasing what it knew [262, 212]. The brain keeps learning after deployment, on two timescales. Within minutes the predictor recalibrates to a new tool or a heavier door (Section 5). Over weeks, practice concentrates on the step that still fails, dopamine scores each attempt and keeps what worked, and consolidation during rest and sleep turns the result into an action primitive that runs with little deliberation (Section 6). Once consolidated, a skill resists interference from the next, so learning adds without erasing.

The survey follows the flow of information through a motor system. It begins with the modalities that carry it, action as output (Section 2) and touch and vision as inputs (Sections 3 and 4), and where brain and robot difer on each. Two models connect them. One predicts the consequence of a command: the cerebellum in the brain, the world model in robotics (Section 5). The other decides what to do: the motor hierarchy from prefrontal to motor cortex in the brain, vision-language models (VLMs) and vision-language-action policies (VLAs) in robotics (Section 6). Each chapter reviews where robotics stands, how the brain solves the same problem, and what to build. Evolution has run this experiment for hundreds of millions of years, and we hope its principles point toward robots that are intelligent, swift, and able to improve themselves.

## 2 Action

![](images/ce2896ffe7e68d9fae17a5950487cb363dd6fd5ea8143f36a72ad03e61c083de.jpg)  
Figure 2: The slot walls supply the rod’s trajectory. The hand pushes along the slot and yields across it, so the walls shape the path and the controller has far less of the curve to compute.

In biology, action leaves the nervous system as spike trains on motor axons; in a robot, it leaves the policy as a vector of numbers. What the limb then does is a transformation of that signal by what lies downstream: muscle, tendon, and joint on one side, servo loops and motors on the other. The deepest diference is what the signal names. Robots command position, where the limb should end up [50]; the nervous system’s final command is efort, how hard each muscle pulls [442, 103]. The brain also sets stifness per task, deciding what must be precise and what can be left to the environment, whereas in a position-controlled robot the stifness is set once by the servo gains, outside the policy’s reach [53, 115]. So a robot must compute every detail of a trajectory, while a human lets the environment supply part of it (Figure 2). We first survey how robotics handles action (Section 2.1), then how biology does it (Section 2.2), and finally draw the inspirations (Section 2.3).

## 2.1 Robotics

Action, what a policy emits, is the boundary between the learned policy and the hand-tuned controllers that drive the motors. In the robotics literature the quantity that crosses it is nearly always a position, handed to a fixed servo stack that turns it into force (Section 2.1.1). The remaining choices concern only how the position is delivered: as a single target or a chunk (Section 2.1.3), and as a discrete token or a continuous number (Section 2.1.4).

## 2.1.1 The policy emits a position; the servo stack supplies the force

The policy issues a position. In nearly all imitation-learning and VLA work the policy outputs a position: joint angles, as in ACT and � [450, 42], or an end-efector pose, as in Difusion Policy and OpenVLA [70, 210]. It runs at 5–50 Hz [70, 42, 209], hands each target to the servo stack and moves on to the next observation; how the target becomes torque is neither observed by the policy nor represented in its output.

The servo stack turns position into force. The conversion is a cascade of nested feedback loops, each output the setpoint of the next, faster loop [357, 173]. The policy’s position target enters a joint PD servo, at up to 1 kHz, that turns position and velocity error into torque via fixed gains [114]. That torque drives a torque loop, which sets the motor current the driver regulates at a still higher rate. This has two costs. A target is correct only under the gains it was trained with, so a policy transfers poorly across actuators unless their response is modeled or randomized [174, 308]. Modeling fits the real actuator’s command-to-torque response and places it in the simulator; randomizing varies gains, latency, friction, and motor strength from one training episode to the next, so the policy cannot rely on any single response and must infer the one it faces from proprioceptive feedback. And the servo cannot tell intended contact from an obstacle:

it pushes with a torque that grows with the error until something yields or a limit trips. Contact-rich behavior must therefore be restored beneath the policy, through force sensing [247], compliant hardware [320], or torque limits [145].

## 2.1.2 Compliance can be added beneath the policy, in hardware, torque, or target

Compliance is the inverse of stifness, the displacement yielded per unit of applied force, and a compliance controller regulates that relation rather than position alone. The robot then behaves like a chosen spring and damper about its target, so a push displaces it by a predictable amount and a contact error produces a bounded force [161, 357]. Three schemes supply the spring, and they difer in where it lives.

Passive compliance puts the spring in the hardware. The remote-center-compliance wrist arranges elastic elements so that a peg touching the chamfer of a hole is steered into alignment by the contact forces themselves, with no sensing or computation [410]. Series elastic actuators place a spring between motor and load, so that motor position becomes a force command through the spring’s deflection [320]. Passive compliance acts at the speed of physics, but its stifness is fixed once, at design time.

Impedance control puts the spring in the torque command. The controller measures the joint’s displacement and velocity from its target, computes the force a virtual spring and damper would exert there, and commands the motors to produce it [161]. Stifness and damping become numbers the controller can change between tasks or within a movement. The price is that the actuators must deliver a commanded torque, through backdrivable transmissions, series elastic actuators, or joint torque sensing, as in the KUKA iiwa and Franka [408, 8, 114]. Conventional industrial arms do not: friction and reflected inertia in their high-ratio gearboxes make motor current a poor estimate of external force [145].

Admittance control puts the spring in the position target. On a stif position-servoed arm the relation runs the other way. A wrist force sensor measures the contact, the controller computes the displacement a virtual spring would sufer under that force, and it shifts the position target by that amount, so the arm yields even though the servo beneath it never outputs a torque [357]. This is how compliance is retrofitted to a position stack, and it is the form the correction of Section 2.3.2 takes.

Only a minority of policies emit something other than position. Compliance control is well established beneath the policy, yet few policies emit anything but position. The exceptions are scattered: a few reinforcement-learned locomotion policies output joint torques directly [67], a few manipulation policies output the force to apply along contact directions [237], and a few emit an impedance target, a position together with the stifness at which to hold it [161, 164].

## 2.1.3 Action chunking decouples control rate from inference rate, at the cost of blindness mid-chunk and seams between chunks

The policy is blind for the executed part of each chunk. Action chunking has the policy predict a chunk of � future targets from one observation rather than one target per step [450, 70]. Fewer decisions per episode means less compounding error, and the control rate is no longer capped by inference speed: a VLA forward pass takes 50–300 ms while joint servos expect targets at 50 Hz or more, so a chunk of 0.5–2 s at 10–50 Hz lets the robot execute one chunk while the next is computed [210, 42]. The cost is that a mid-chunk disturbance is answered only by the servo stack of Section 2.1.1, which holds the commanded positions against it whether or not that is appropriate.

Consecutive chunks must be joined. Each chunk is generated from an observation that is stale by the inference time, and nothing forces it to agree with the chunk still executing, so the boundary can produce a discontinuity. Temporal ensembling averages the overlapping predictions of successive chunks [450]; bidirectional decoding samples several candidates and keeps the one most consistent with the executed prefix [250]; real-time chunking freezes the actions that will execute during inference and generates the rest conditioned on them, with a soft mask so that later actions stay free to respond to the new observation [43].

## 2.1.4 Token or number, absolute or delta, the representation changes how the position is encoded

Discrete heads inherit the language prior; continuous heads give precision and multimodality. Binning each action dimension into a fixed vocabulary lets a pretrained VLM emit actions by next-token prediction [49, 210], but a chunk binned step by step is mostly redundant, since successive samples of a smooth trajectory are nearly equal; FAST compresses chunks with a discrete cosine transform before byte-pair encoding [310]. Continuous heads instead regress the numbers directly: $L _ { 1 }$ regression is cheapest but collapses several valid actions onto a single compromise value [70], and difusion or flow matching recovers the full distribution over several denoising steps [159, 70, 42]. �<sub>0</sub> and �<sub>0</sub>-FAST put the two heads on a shared backbone and isolate exactly this trade-of [42, 310].

Delta actions pool across embodiments; absolute targets stop error compounding. Delta end-efector actions [48, 294, 210] are relative to the current pose, hence invariant to initial configuration and small enough to tokenize well, which is why cross-embodiment datasets favor them [296]. But each delta is applied to wherever the arm actually is, so a lag or slip shifts every later target with it [70]. Absolute targets, joint angles in ACT and $\pi _ { 0 }$ or poses in Difusion Policy [450, 42, 70], live in a fixed frame, so each error is corrected at the next step rather than carried forward [70]. But that frame is tied to one body: joint angles to one arm’s kinematics, poses to one base placement.

## 2.2 Biology

We define action in biology as the output of the motor neurons of the spinal cord, the final common path through which every descending and reflex influence must pass [347]. The brain’s decisions arrive there as synaptic input, and what leaves is a train of action potentials that the muscle turns into movement [197, 442].

## 2.2.1 The motor neuron integrates cortical, brainstem, and reflex drive into one command

Every alpha motor neuron receives a voluntary stream from cortex, a postural and locomotor stream from the brainstem, and a reflex stream from sensors in the muscle itself, and its output depends largely on their sum. The cortical stream is on the order of $1 0 ^ { 6 }$ corticospinal axons [226, 225], most ending on spinal interneurons, that drive on the order of $1 0 ^ { 5 }$ motor neurons [378]. The brainstem stream carries posture and the go and speed commands for locomotion; the rhythm itself comes from central pattern generators, in the spinal cord for walking and running and in the brainstem for breathing and chewing, which produce it on their own [139, 351]. The reflex stream closes fast loops that never wait on the brain: the spinal stretch reflex returns a limb toward its load within roughly 30 ms, with a cortical component following 20–30 ms later [261], and the grip reflex of Section 2.2.4 tightens the fingers on a slipping object. The action that leaves the nervous system is therefore the sum of the brain’s decision, locally generated rhythm, and sensory feedback.

## 2.2.2 The population firing rate of a muscle’s motor neurons is translated into efort

More drive onto a muscle’s motor pool recruits more of its neurons and fires each faster [153, 273]. The muscle turns that firing into a contraction that is its smoothed copy [306], so the command is an efort. That force rises with the muscle’s stretch over much of its working range, falls with its shortening speed [158, 132], and acts through a lever arm that changes with joint angle [281], so the same firing gives a diferent torque at every posture, outside neural control. The stretch dependence makes an active muscle a spring, and the limb settles wherever its pull and the world agree.

## 2.2.3 Opposing muscle pairs set torque and stifness separately

Muscles come in opposing pairs, one pulling each way across a joint, so every joint is held between two springs. Two drives per joint thus set two quantities at once [160]. Their diference is the net torque (Figure 3, left): flexor pull above extensor pull flexes the elbow, the reverse extends it, and equal pulls leave it balanced. Their sum sets the stifness (Figure 3, right): co-contracting both muscles leaves the pose and the net torque unchanged, but the elbow that was easy to push aside under light co-contraction now resists the same push under strong co-contraction. Torque and stifness are therefore two largely independent commands sent through one pair of muscles.

![](images/81e71f1d8a214235289f22c21a8f598956eea3564076d159e88192fbf4c8e5b5.jpg)  
Figure 3: Opposing muscles set torque and stifness separately. The diference of flexor and extensor pull sets the net torque, and their sum sets the stifness at the same pose.

## 2.2.4 Slip during a grasp is caught by a reflex that corrects grip force, with no position in the loop

During a grasp, grip force follows the motor pool’s firing with a short delay [187, 306], and the whole task is specified in that quantity. The precision grip regulates the ratio of grip force to load force, holds it a fixed margin above the slip limit, and raises it 74 ± 9 ms after a micro-slip [189], with no positional quantity anywhere in the specification. The slip itself is read from the skin’s mechanoreceptors, whose classes and the signals they combine into a slip event are described in Section 3.2.1. The correction is a reflex rather than a decision: a sudden pull on the held object triggers the same grip increase within about 70 ms, before any voluntary response [192], and anesthetizing the fingertips delays and weakens it, so the loop is closed mainly by those cutaneous aferents rather than by muscle sense [191]; Section 3 takes up what the loop should look like in a robot.

## 2.3 Inspiration

From the two subsections above we draw two lessons for the policy interface. The first is that compliance control matters because it lets the world supply parameters the policy would otherwise have to compute (Section 2.3.1). The second is our proposal for achieving it on the arms most policies run on, which accept only position targets: keep the position output and add a stifness beneath it, with a touch sensor measuring the contact so the controller can take from it where to yield and how hard to press, moving the position target beneath the policy (Section 2.3.2).

## 2.3.1 Use compliance so the world supplies what the policy would otherwise compute

Consider carrying a rod through a curved slot (Figure 2). A robot with only position commands has to perceive the precise shape of the whole curve, and any error at contact becomes uncontrolled force through a stif servo. A human does not track the curve. The hand pushes with high stifness along the slot and low stifness across it, and lets the board do the shaping: as long as the push is roughly along the slot, the wrist tolerates some wobble, and whatever the wall presents is fine, because the hand yields to it. The same holds when a yielding wrist scrubs a bumpy surface (Figure 4).

The controller’s only job is to choose which axes to hold and which to yield; the curve of the slot and the height of each bump arrive through contact for free.

![](images/705c6607e04f9f3a901d7f2de1923342289d40de1c338f463a8681fc9998e62c.jpg)  
Figure 4: Contact supplies the surface detail. A yielding wrist and a soft towel follow the bumps, so the controller computes fewer corrections.

## 2.3.2 On the position-controlled arms most policies run on, keep the position output, add a stifness beneath it, and let touch set where to yield and how hard to press

We propose that the policy, like the muscle pair of Figure 3, command a stifness alongside its target, and that the stifness be realized beneath it: by impedance control where the joints deliver torque, and by admittance on the position-controlled arms most policies run on, where even Adaptive Compliance Policy had to recover its stifness through a wrist force sensor [164]. We take the position-controlled arm as the main case and let touch set where to yield and how hard to press.

Touch tells the controller where to yield and how hard to press. On a position-controlled arm we propose adding the stifness as admittance (Section 2.1.2): a touch sensor measures the contact and the controller moves the position target in response. Against a hard wall, the direction of the sensed force is the direction to yield in, and the rest of the commanded motion goes through unchanged. Against a soft object, the magnitude decides when to stop closing. This rule can be written as code by a language model, as in Code as Policies [239], or it can be learned. OmniTacTune [437] keeps the visual policy frozen and adds a small residual network, trained by reinforcement on the real robot, that reads the tactile signal and corrects the end-efector target while each chunk executes, with the residual scaled to 5–15% of the base action. It raises success on contact-rich tasks from 5–40% to 85–100% within 40–80 minutes of training, on an arm that never outputs a force.

## 3 Touch

![](images/0a5a5f1509299ee7cf10297664191171a5c711e46cb98372c67afb830206c7f9.jpg)  
Figure 5: Touch alone finds the handle and opens the door. With eyes closed the hand feels its way to the handle, turns it and pushes the door open; in the dark a camera has nothing to ofer.

A human can find a door handle in the dark and open the door without a single glance (Figure 5). Most current robot policies cannot, because they rely on vision [42, 123, 255]. One might argue that, with the lights on, vision could cover what touch does, but the human evidence says otherwise: with anesthetized fingertips and the hand in plain view, subjects still pick up some objects, yet they apply excessive, poorly scaled forces and drop others [191, 276, 21].

Beyond empirical evidence, a first-principles view says the same. Touch supplies the force in Newton’s second law, $F = m a ;$ vision sees only displacement, velocity, and acceleration, the right-hand side, and can infer force from them only indirectly. A robot that senses only motion sees the efect of every interaction and must guess its cause, and that cause is what physical intelligence must reason about. This section examines how humans use touch and what robotics can learn from it.

## 3.1 Robotics

Tactile sensing has recently attracted growing attention as an input modality for robot learning [255]. We first discuss sensors and then how the policy consumes their signal.

## 3.1.1 No sensor family matches human skin on resolution, speed, and force at once

Every robotic touch sensor is a physical filter. The transduction principle fixes what the skin can resolve in space, how fast it responds, and which components of force it reports, and no current family scores well on all three [85, 255, 424]. Table 1 shows the pattern: camera-based pads resolve tens of microns, piezoelectric films catch kilohertz transients but are blind to steady pressure, and the cheapest arrays read normal force only [85, 165, 373].

Durability is a further limit. Touch sensors sit on the part of the robot built to collide with the world, yet the soft elastomer that makes a sensor sensitive wears out long before the electronics beneath it [220, 37].

<table><tr><td>Sensor type</td><td>How it works</td><td>Spatial res.</td><td>Speed</td><td>Strengths</td><td>Weaknesses</td></tr><tr><td>Piezoresistive [85, 169]</td><td>Pressure squeezes a conductive polymer and lowers its resistance</td><td>mm-scale</td><td>tens of Hz</td><td>Cheap, thin, flexible; scales to large arrays</td><td>Hysteresis; reading drifts with load history; normal force only</td></tr><tr><td>Capacitive [85, 434]</td><td>Pressure narrows the gap between two electrodes (touchscreen principle)</td><td>mm-scale</td><td>~100Hz</td><td>Sensitive to light contact; mature technology</td><td>Picks up electrical interference; mostly normal force</td></tr><tr><td>Piezoelectric [165,85]</td><td>A polymer film emits a voltage when strained</td><td>mm-scale</td><td>kHz</td><td>Captures vibration, texture, and the onset of slip</td><td>Responds only to change; blind to steady pressure</td></tr><tr><td>Barometric [373]</td><td>Air-pressure chips cast inside rubber, one chip per taxel</td><td>mm-scale</td><td>&gt;100 Hz</td><td>Robust, inexpensive, factory-calibrated</td><td>Coarse; normal force only</td></tr><tr><td>Magnetic [37,379]</td><td>Magnetized elastomer over magnetometer chips; deformation shifts the field they read</td><td>mm-scale</td><td>hundreds of Hz</td><td>Three-axis force including shear; thin; replaceable skin</td><td>Hysteresis; signal drifts as the elastomer ages, so each new skin needs recalibration</td></tr><tr><td>Multimodal fluid-filled fingertip [111,409]</td><td>Fluid-filled fingertip with impedance electrodes, a pressure transducer and a thermistor</td><td>tens of sensing points per fingertip</td><td>~100 Hz, with a kHz vibration channel</td><td>Pressure, vibration, and heat flow together, the closest to a human fingertip</td><td>Expensive; signals need learned decoding</td></tr><tr><td>Vision-based [441, 220, 405]</td><td>A small camera films an illuminated gel or pin array deforming under contact</td><td>tens to hundreds of µm</td><td>camera rate, 30-60 Hz</td><td>Spatial resolution finer than human skin; full contact geometry, shear from marker motion, slip onset</td><td>Bulky; slow; needs image processing; gel wears out</td></tr></table>

Table 1: No single sensor family combines fine resolution, kilohertz speed, and shear. Resolution and speed are order-of-magnitude values typical of each family, and the cited works are representative examples. A taxel (“tactile pixel”) is one sensing point on the skin.

Touch rarely extends beyond the fingertips. Most sensorized hands carry touch only at the fingertips, as caps or gripper pads, and leave the phalanges and palm bare [220, 255]. Even at the tip, the sensor covers the pad and not the sides of the fingers that steer chopsticks and scissors [85, 424]. Whole-hand skins remain rare: the scalable ones read normal force only [367], and shear-capable coverage of palm and fingers exists in a few research hands [379, 451].

## 3.1.2 Policies now take touch as an input, but data and control rate hold it back

Touch enters the policy as an extra image or as learned tokens. The design choice is how the tactile signal is represented. Because vision-based sensors emit images, the simplest approach treats the tactile image as an additional camera view and reuses the vision encoder [149]. Touch-specific pretraining improves on this borrowed encoder, either through self-supervised learning on tactile video or through contrastive alignment of touch with vision and language [155, 425]. Other policies learn tactile tokens of their own: a point cloud from piezoresistive pads [169], a single token from the force history [170], latent tokens from a dedicated tactile corpus [288], or a few tokens compressing the tactile history [404].

Cost, standardization, scale, and speed still hold touch back. Tactile sensors are expensive and wear quickly, so most demonstrations are collected without them. Each sensor family emits a diferent signal, an image, a three-axis force, or a pressure array, so data from diferent platforms cannot be pooled without a learned shared latent space [108, 305]. Tactile corpora are therefore orders of magnitude smaller than video [432, 288]. Finally, policies run at camera rate, 10–60 Hz, and miss the hundreds-of-hertz transients that signal events such as slip (Table 1).

## 3.2 Biology

## 3.2.1 Four receptor classes encode touch in spike timing

![](images/e00e6410890af3229e7fc270ce734ea5db5b13b36af4318a0c118e6634d0ed35.jpg)  
Figure 6: Slowly adapting (SA) aferents report steady pressure and rapidly adapting (RA) ones report change, so the four classes split what touch carries. The traces are schematic responses to a ramp-and-hold skin indentation: SA aferents sustain firing during the hold, RA aferents respond transiently during pressing and release, and SA2 aferents also fire at a low rate before contact, driven by resting skin stretch. RA aferents can also fire during ongoing motion or vibration. Spike counts and timing are illustrative rather than measured; responses depend on the stimulus and aferent [194, 2].

A mechanoreceptor reports only through spike timing. Touch starts in the skin. Buried at diferent depths are the endings of sensory nerve fibers, most ending in a specialized structure of tissue: the mechanoreceptors [186, 194, 2]. When the capsule deforms, the nerve ending fires brief, stereotyped electrical pulses called spikes. A spike cannot be made bigger to say “more”; a stronger press makes the ending fire sooner and more often, so everything a fiber reports is carried in when its spikes come and how often [3, 194] (Figure 6).

Tactile percepts arise from combining the four receptor classes. A receptor’s wrapping and depth set two properties: how large a patch of skin it listens to, and whether it reports steady contact (slowly adapting, SA) or only change (rapidly adapting, RA). The four combinations give the four classes in Table 2: SA1 reports sustained pressure and fine shape, RA1 the onset of movement, SA2 skin stretch and force direction, and RA2 high-frequency vibration. None of them reports a tactile feeling on its own. Texture, weight, and slip each arise from combining the four channels [188]. When a held glass begins to slide, RA1 aferents fire the most reliable burst at the onset of the slip, with RA2 and SA1 aferents contributing, while the grip force itself has not changed [189]: a burst of change with no change in squeeze signals slip.

<table><tr><td>Receptor</td><td>Where it sits</td><td>Skin patch it hears</td><td>Steady or change?</td><td>Everyday role</td></tr><tr><td>SA1 (Merkel ending)</td><td>Just under the surface; densest at the fingertip</td><td>Tiny (2–4 mm), sharply bounded</td><td>Keeps firing under steady pressure</td><td>Reading fine shape and texture; edges and curvature, the Braille channel</td></tr><tr><td>RA1 (Meissner corpuscle)</td><td>Just under the surface, in the hairless skin of the hand</td><td>Small (3–5 mm)</td><td>Fires only when contact changes</td><td>Catching micro-slips and low-frequency flutter, the trigger for tightening your grip</td></tr><tr><td>SA2 (Ruffini ending)</td><td>Deeper in the skin; sparse</td><td>Large, blurry borders</td><td>Keeps firing</td><td>Sensing skin stretch, finger posture, and which way a force is pulling</td></tr><tr><td>RA2 (Pacinian corpuscle)</td><td>Deepest of all; sparse, roughly 350 per finger</td><td>Very large, a whole finger or the hand</td><td>Fires only at sharp changes</td><td>Feeling high-frequency vibration through a held tool, pen on paper</td></tr></table>

Table 2: Four aferent classes split touch by receptive-field size and adaptation [186, 389, 194]. The pairing of SA2 aferents with Rufini endings is disputed, since the human fingertip holds few such endings [303]. Figure 6 uses the same codes.

## 3.2.2 Successive contacts build an object representation beyond geometry

S1 estimates the surface under one fingertip. Discriminative-touch signals from a fingertip reach the primary somatosensory cortex (S1) through the dorsal column–medial lemniscus pathway (Figure 8) [197]. The input is dense, roughly 17,000 tactile aferents for the glabrous skin of the hand [186, 78], and S1 turns it into a first estimate of the surface under each fingertip: area 2 integrates cutaneous input with finger posture to recover the curvature and orientation of the surface in contact [197].

Parietal cortex assembles the whole object from many contacts. From S1 the signal passes to the secondary somatosensory cortex (S2) and on to posterior parietal cortex, where processing moves from individual skin contacts to the object. Parietal cortex relates each contact to hand and arm posture, placing it in space, and so combines touches made by diferent fingers at diferent times into an estimate of the object’s three-dimensional form [197].

Vision and touch converge into one object representation. Tactile and visual shape processing converge in regions such as the lateral occipital tactile-visual area (LOtv), which responds to objects explored by touch as well as those seen [11]. The two estimates are weighted by their reliability: clear vision dominates, and touch gains weight as visual information becomes uncertain [104]. The result is one object representation that combines visible geometry with properties only contact reveals, such as weight, texture, and compliance (Figure 7).

![](images/12079ea032e547501f82416c4ba8fd86fc0f41f528da6b924022d5fa4ed0b84b.jpg)  
Figure 7: Vision and touch build a single object model, and each sense fills in what the other cannot. Vision covers the visible surfaces, while touch supplies weight, texture, slipperiness, and compliance, and explores the occluded back.

## 3.2.3 Tactile reflexes correct grip within about 75 ms, faster than any visual correction

![](images/195699befffc380293aa2e27171e8e8f6609377758b5e197250d119a3f4d8dfa.jpg)  
Figure 8: Three neurons carry a fingertip signal to primary somatosensory cortex (S1), a chain short enough to leave time for a tactile correction within about 75 ms. The first-order neuron ascends on the same side through the spinal dorsal column to the cuneate nucleus of the medulla; the second-order neuron crosses there and ascends to the ventral posterolateral (VPL) thalamus; the third-order neuron projects to S1 on the side opposite the stimulated hand. Each line ends in a dot at its synapse, and the legend colors mark the three cell bodies. Other branches and circuits are omitted [197].

Touch does not have to wait for a full object representation before it acts, because tactile signals reach the motor system along two pathways of diferent speed. The reflex pathway runs through the spinal cord and brainstem, or through the chain of Figure 8 to S1 and the neighboring motor cortex: when a held object begins to slip, grip force rises $7 4 \pm 9$ ms later, before the slip reaches awareness [189, 190, 188]. The deliberate pathway passes through conscious perception and planning and takes roughly 150 ms or more, the range of voluntary reaction times [188]. The reflex pathway is what vision cannot match, since even the fastest visual corrections take roughly 100 ms or more (Section 4.2.5).

## 3.2.4 The brain predicts touch and acts on the discrepancy

Why can you not tickle yourself? Because the brain predicts how your own action will feel: a copy of every motor command is used to forecast the touch it should produce [44, 412, 112]. The diference between predicted and actual touch is the control signal: either the prediction was of or the execution fell short. It serves two purposes at once. It triggers a correction of the action in progress [187], and it updates the predictor, a role attributed to the cerebellum [413], so the next lift starts with nearly correct grip forces. We return to the cerebellum, and to what it shares with a robot world model, in Section 5.2

## 3.2.5 Descending control suppresses predictable and irrelevant tactile channels

![](images/6d44ed7c092ee14198b1fc985287e5decb453753cf123339cf484291a75bcb6f.jpg)  
Figure 9: Descending signals from cortex turn down predicted or irrelevant tactile channels and turn up the task-relevant one, so the fingertip signal arrives at high contrast.

Prediction and attention together decide what the tactile system transmits. The skin streams signals constantly (clothes, watchband, chair), yet by a textbook estimate, of roughly $1 0 ^ { 6 }$ bits/s entering the nerves only about five bits/s reach perception [456]. The selection is top-down: descending signals set the gain at each relay, turning down channels that prediction marks as expected, such as touch from your own movement, or that attention marks as irrelevant, and turning up the relevant one [61, 377, 411]. Filtering this early spares downstream spikes, which are costly [20, 231], and delivers the attended contact to cortex at high signal-to-noise (Figure 9).

## 3.3 Inspiration

Touch should measure the contact force that compliance control needs (Section 3.3.1), reveal the physical properties of objects (Section 3.3.2), and gather evidence actively with small probes (Section 3.3.3).

## 3.3.1 Represent touch as a force field first, and let it drive compliance control

We propose that touch enter the model as a force field: a contact force per point on the skin, with normal and shear components. Most sensor families can be calibrated to this form, at least approximately [441, 37], so data from diferent sensors pool directly and one embedding serves them all, where cross-sensor encoders must first learn a shared latent space [108, 305]. The force field keeps only force and discards texture, vibration, and temperature. Raw signals keep all of these and also work [155, 288], but each sensor then needs its own tokenizer and enough data to train it.

The model uses that representation in two ways, both through the contact force in strength and direction. Against a fragile or soft object, the strength decides when to stop closing; against a hard surface, the direction marks the axis to yield along, the compliance rule of Section 2.3.2. Either form of the model’s action plan can apply it (Section 6.3.3): as code, the model writes the rule, yield along the sensed normal and stop at a force threshold [239]; as a reactive policy head, the head takes the tactile tokens as input and corrects the trajectory at the control rate.

## 3.3.2 Train the model to predict the tactile consequence of each command

Given past observations and a new command, the model should predict the sensory consequence [156, 249, 443]: if past touch suggested high friction, how far the object slides under a given push; if it suggested a soft sponge, how much the sponge deforms under a given grip. This forces the model to infer the properties that geometry [304, 206] leaves out and contact measures directly: mass, friction, compliance, and fragility, on which grip force and the risk of slip or breakage depend [432, 227]. We train on exactly such data: the robot’s own executed actions, command as question and tactile outcome as answer, free of annotation, through the force-field calibration of Section 3.3.1 so one model serves every sensor.

The same prediction pays of at run time. Once the real touch arrives, the model compares it with what it expected and acts on the diference, as the brain does (Section 3.2.4). A grip that slips or a surface that gives more than predicted shows up as a mismatch at the first contact, and the sign and size of that mismatch say which action compensates: tighten the grip, ease the press.

## 3.3.3 Probe actively to test hypotheses about properties that cannot be seen

When the property that matters cannot be seen, the robot should create the evidence before committing: lift the object a few millimeters to estimate load, tilt it to feel whether anything shifts inside, or press gently to measure stifness, and let the outcome set the following strategy (Figure 10) [45, 227]. Probes should be uncertainty-driven, choosing the contact where competing hypotheses predict diferent outcomes [258, 110, 133]. The same probes are the cheapest self-improvement data: each is an action with a measured consequence, so it trains the model’s expectation of what its actions will do, data that passive demonstrations rarely contain (Section 5.3.1) [127, 316].

![](images/92e3f5ad8caa2bc85deae4cacf2d80e4116d48ba4ed5fd9ee7dc166e64fe5b19.jpg)  
Figure 10: When an object’s properties cannot be seen, the robot should probe before it commits. Lifting, tilting, and pressing within safe bounds each test a hypothesis the camera cannot settle.

## 4 Vision

![](images/3fdbf967cfacfaf06d19e900ec506a2e266bffdb095fbc7b9474e17d2d2798c1.jpg)  
Figure 11: The same ball is read twice, by the two visual streams in the brain. The ventral stream resolves its pattern and identity; the dorsal stream resolves its trajectory and guides the catch.

Robotics borrows its vision encoders from computer vision, where they were optimized for object recognition and for image or video generation. An agent acting on an object does not need every detail of it at every instant; it needs where the object is, how it moves, and how to take hold of it. Biology runs two parallel “vision encoders” (Figure 11): a slow one for fine detail and recognition, and a fast one dedicated to guiding motor behavior. Current encoders resemble the first, and the second has no counterpart. We first survey what current encoders were built for (Section 4.1), then turn to how the brain builds a second visual system for motor behavior (Section 4.2), and finally draw the algorithmic inspirations (Section 4.3).

## 4.1 Robotics

We review the training objective for current vision encoders (Section 4.1.1), how the backbone behind them reshapes their tokens (Section 4.1.2), and the limits that follow (Section 4.1.3).

## 4.1.1 The pretraining objective decides what the tokens keep

Almost all tokenizers are ViTs [94] pretrained at scale and then frozen or lightly fine-tuned; what distinguishes them is the pretraining objective, which leaves a durable fingerprint on the representation. Nearly every recipe asks for one of two things: predict the image or video itself, in pixels or in an embedding of it, or align it with a text description.

Self-supervised encoders predict the input, in pixel space or in latent space. The recipes difer mainly in where the prediction target lives. Masked pixel reconstruction puts it in pixel space: the model repaints masked patches directly [151, 419, 27, 380]. Masked latent prediction instead forecasts an exponential-moving-average teacher’s embeddings of the masked regions [17]; on video this recovers motion and temporal structure without paying for pixel fidelity [28, 18]. View-invariant self-distillation targets the teacher’s embedding of a diferent augmented view of the same image, with centering and sharpening to prevent collapse [55, 299, 359]; correspondence, segmentation, and depth emerge in frozen probes.

Caption alignment buys semantics but never asks for geometry. Image–text contrastive learning predicts which caption belongs to which image, pulling matched pairs together in a joint embedding space [325, 445, 385]. Because language alignment is the objective itself, these encoders became the vision towers of VLMs and, through them, of most

VLAs. The cost is inherited from the captions: they rarely mention exact pose, contact, or metric layout, so nothing in the objective pressures the encoder to represent geometry.

## 4.1.2 The VLM or video world-model backbone acts as an extended vision encoder

The backbone that consumes the tokens decides what the action head finally reads. In a VLA, image tokens enter the language model through a projector; the action head then reads its hidden states or key-value cache [42, 40], or the language model emits the action tokens itself [210], so vision arrives fused with the instruction and the backbone’s semantic knowledge. World Action Models replace the language model with a video-difusion backbone, so the action head reads features shaped by predicting scene change (Section 5.1.1) [168, 431, 101]. Smaller imitation policies feed encoder features straight to the action head [70, 450, 102]. In every case, recognition, language, or generation shaped the representation, and none of them is control.

## 4.1.3 Current vision stacks are slow, blind to velocity, and miss fine detail

Policies act on one stale observation for up to seconds. Cameras deliver 30–60 frames per second. Autoregressive VLAs typically infer at a few hertz [49, 210], and even the fastest flow-matching VLAs, which infer in tens of milliseconds [42, 257], still commit half a second to two seconds of motion to one observation and execute it blind [370]. Widening the window multiplies the tokens the backbone must process [438, 251], so most VLAs act on a single stale frame [65, 381].

Single frames carry no velocity. Any event shorter than the hundreds of milliseconds a chunk executes between looks, such as the onset of a slip (Section 3.2), falls between frames, and a frame holds no motion to recover it from. Image-pretrained encoders represent task progression poorly even with temporal features added on top [383]; Section 4.3.1 takes up the video and latent-prediction remedies.

Fine detail is lost. Thin cables, small screws, and fine texture span a few pixels at the 224–384-pixel inputs these encoders take [94, 445]. The limit is not the encoder: models that take higher resolution [385, 359, 23] or reconstruct the image [151, 299] keep such detail. The limit is the token budget: tokens grow with the square of the resolution, and a VLA multiplies them by its cameras and its time window, so it runs the encoder small and loses the detail the task needs.

## 4.2 Biology

This section traces how primate vision splits into a stream for recognition and a stream for action, then follows the second.

## 4.2.1 The retina compresses before transmitting, and recognition and action get separate streams

The retina funnels the output of roughly 100 million photoreceptors [83] into about one million ganglion-cell axons [82], a hundredfold compression forced at the sensor itself. From there the signal takes two routes. Most of it relays through the lateral geniculate nucleus of the thalamus to primary visual cortex (V1) [197], and by the time it arrives it is already features, not pixels: V1 neurons are selective for oriented edges, binocular disparity, and, crucially for a moving animal, motion direction and speed. A smaller fraction takes an older, subcortical route straight to the superior colliculus, which drives orienting movements of the eyes and head at very low latency and without awareness [197, 415].

Recognition and control get separate streams with diferent codes. Beyond V1, primate vision divides into two anatomically distinct streams (Figure 12) [388, 129, 272]; a third, lateral pathway for social perception has been proposed [317]. The ventral stream, into inferotemporal cortex, serves recognition. Its codes are position-invariant and durable: “a mug, near the laptop” survives eye movements and can be stored in memory. The dorsal stream, into posterior parietal cortex and on to premotor and motor areas, serves control. Its codes are metric and anchored to the hand or eye: “handle 31 cm out, 40<sup>◦</sup> from the wrist”, recomputed continuously and short-lived. Picking up that mug needs both: the ventral stream supplies what it is and which part to take hold of, the dorsal stream where the handle is relative to the hand right now. How cleanly the two streams divide recognition from control is contested [339], but that debate is not discussed in this paper. The rest of this section follows the dorsal stream.

![](images/9ace9cb1ee8deb74df53fc2e08f9f590773d3d187f569567341b4d01b7f2b67c.jpg)  
Amber: collicular route, bypassing cortex. Gray: shared early pathway.

Figure 12: Vision for action and vision for recognition separate after V1, and a collicular route bypasses cortex entirely. The geniculate route through V1 divides into a dorsal stream through posterior parietal cortex to premotor and motor areas (vision for action) and a ventral stream through V2 and V4 into inferotemporal cortex (vision for recognition); a separate collicular route drives fast orienting of eyes and head without cortex.

## 4.2.2 The dorsal stream is modeled as a bank of parallel channels, one per class of action

In the prevailing model, the dorsal stream is organized by action rather than by feature: each channel pairs a region in parietal cortex that estimates one aspect of the world with a partner in premotor, motor, or oculomotor areas that plans the matching movement (Figure 13) [77, 331]. The tuning of each area below is well replicated; the division into channels is the model’s interpretation of it.

• Motion: what is moving, and how? The middle temporal area (MT) measures the direction and speed of movement at each point in the image [46]. Its neighbor MST combines these local measurements into whole-field patterns, the biological counterpart of optic flow: expansion, which signals approach and gives time-to-contact [228]; translation, which signals self-motion and gives heading; and rotation [97].

• Reach: where must the hand go? Areas V6A and MIP encode the direction and depth of the reach target, and dorsal premotor cortex turns that estimate into an arm trajectory [121, 30].

• Grasp: how must the hand be shaped? Area AIP extracts the size, shape, and orientation of the graspable parts of an object, and its premotor partner F5 selects a matching grip type, such as a precision pinch or a whole-hand grasp [280, 279, 338].

• Orient: how must the hand be turned? Orientation for action appears to be computed separately from perception. Patient D.F., with ventral-stream damage, could not report the orientation of a slot, even by turning a card to match it, yet she posted the card through it, rotating her hand correctly during the reach [131]. Neurons in V6A are tuned to wrist orientation during reach-to-grasp, so this computation sits between the reach and grasp channels [106].

• Gaze: where to look next? Area LIP keeps a priority map of the scene, combining what stands out with what the task needs, and the eyes move to its peak [39].

• Near-body space: what is close to the body? Area VIP and its premotor partner F4 track objects in the space within arm’s reach of the head and body, for avoidance and close-range reaching [113, 137].

• Tool use: what is in the hand? A tool in use appears to join the hand’s channel: after a monkey uses a rake, parietal neurons that respond to touch on the hand and to objects seen near it extend their receptive fields along the tool [177]. F5 grasp neurons follow the tool tip even with reverse pliers, which close when the hand opens [387].

The bank of channels is a model built mostly from macaque recordings, lesions, and stimulation [331], and how separate the channels are is debated. The circuits are densely interconnected, a reach area also codes wrist orientation [106], and the dorsal and ventral streams exchange object information rather than working apart [272, 339]. Its transfer to humans is also incomplete: the same parietal maps have expanded more than the rest of visual cortex [267], and a left parietal region responding to tool use has no counterpart found so far in monkeys [307, 198].

![](images/cc87625d543b0c4d609dd00d61e8b479ea545df3931a6ed1bcbe2d222c72bcc4.jpg)  
Teal: dorsal-stream visual areas. Orange: their motor partners. Labels name each area's dominant contribution; the circuits are interconnected. Orient and tool use are not separate areas: hand orientation is read in V6A beside reach, and a tool in use extends the grasp channel.

Figure 13: In the prevailing model, the dorsal stream is a bank of parallel channels. The figure expands the dorsal branch of Figure 12 and draws the motion areas MT and MST, which that figure folds into the parietal box, as a separate stage. Each visual area estimates one quantity the body needs (where to look, what is moving, what is within reach, where to reach, how to turn and shape the hand, and what a tool in hand reaches) and sends it to a partner in premotor, motor, or oculomotor areas; the italic line in each visual box names the reference frame the channel works in (Section 4.2.4). Each label names an area’s dominant contribution rather than its only one [331].

## 4.2.3 Vision selects one object, holds it on the fovea, and tracks its motion

The dorsal channels do not process the whole scene. The fovea resolves fine detail over a degree or two [83], so the eye samples the scene in three or four fixations a second, placed by the task [429, 223]: in manipulation they land where the hand will make contact and move on before it arrives [221, 193, 150]. Selection precedes the eye movement: attention shifts to the target before the saccade [88], and motion neurons in MT and MST respond to the selected object while nearby motion is suppressed [382, 328], so the channels of Section 4.2.2 report that object alone. Once selected, the object is tracked: smooth pursuit holds it on the fovea, so its motion is read from the eye command rather than from retinal slip [242, 222].

A few other moving objects are tracked at the same time, but only as unlabeled positions: attention keeps an index on each one as it moves, enough to find it again or to notice that it is about to collide, while what it is remains unresolved until the fovea visits it [323, 268, 162].

## 4.2.4 Each dorsal channel re-expresses visual information toward the frame its efector needs

![](images/4223ea97ab91b5bea6920db7875ab314f721dc311a20ad41cade9b5c4c980750.jpg)  
Figure 14: A visually guided action must re-express the retinal report in the frame its efector needs. The same dumpling has diferent coordinates in each frame along the chain: eye, head, body, hand, tool, and object.

The frame follows the efector: eye for gaze, head for near space, hand for reach and grasp. The retina reports everything relative to the eye, but the eye moves in the head and the head on the body, and an action needs the target relative to the body part that will act on it (Figure 14). This re-referencing underlies every visually guided movement [13, 80]. Gaze needs no transform (Figure 13): an eye movement is an eye-centered vector, so the motion and gaze channels stay eye-centered [46, 77]. Near-body space is coded relative to the head and limbs, so an approaching object is located against the face or arm it may hit [99, 136]. Reach targets are coded in frames that range from eye-centered to hand-centered across parietal areas [30, 121]; grasp areas code the size, shape, and orientation of the graspable part, an object-centered description [335, 280]; and a tool in use extends the hand frame to its tip (Section 4.2.2). The assignment is flexible: human reach and grasp areas code the same target in a gaze-centered or a body-centered frame depending on whether it is seen or felt [34, 232], and they carry its direction and distance in several frames at once [54].

The transform is a population code built from gain fields. Each channel computes its frame from retinal input using gain fields [14]: a parietal neuron keeps its retinal receptive field, but eye or head position scales its response. Few single neurons encode the target cleanly in the new frame; the population forms a basis from which downstream areas read out any frame they need as a linear combination [318]. The frames in Figure 13 are therefore idealizations: measured receptive fields are commonly mixtures that shift only partway between eye and head coordinates [22].

## 4.2.5 Vision is slow, so the brain acts on a prediction and corrects it later

Vision is slow: recognizing what a scene contains takes about 150 ms [375], and a visual correction to a reach begins 100–150 ms after the target moves [319], long enough for a reaching hand to travel several centimeters. The brain therefore predicts how the image will change and uses the delayed image only to correct the prediction. The image changes for three reasons, ordered here from easiest to predict to hardest.

The eye moved: an exact command. Each eye jump shifts the whole image. A copy of the eye-movement command reaches the visual areas before the eye moves, and neurons there shift the patch of image they watch to where it will land [98, 361]. The scene estimate is updated ahead of the movement, one leading account of why the world appears stable across eye movements [415].

The body moved: a command plus limb dynamics. During a reach, visual feedback about the hand lags on the order of 100–200 ms, so a controller steering on feedback alone would always correct toward where the hand was [87]. The cerebellum runs a forward model instead: a copy of the arm command predicts the limb’s current state, and delayed vision corrects the prediction, the structure of a Kalman filter [412, 269, 196]. The correction runs below awareness: a target displaced during an eye movement is never seen to jump, yet the hand corrects its path [130, 319].

The world moved: no command, only observation and priors. A thrown ball’s motion must be extrapolated from what is seen. Perceiving where a moving object is belongs to the visual pathway. The retina already fires ahead of a moving stimulus [35], and a moving object is perceived ahead of a flash at the same position, which one account attributes to continued extrapolation in cortex, although the mechanism remains debated [290, 386]. Learned physics fills the gaps: the cortex holds an internal model of gravity [176]. Timing an action on a moving object brings in the cerebellum, which keeps tracking a target behind an occluder and predicts when it will reappear [59, 300]. Catching needs both: the grip is timed from the gravity model, so firmly that astronauts keep applying it in orbit [264].

## 4.3 Inspiration

We make three proposals for robotic vision. First, add a fast encoder built for streaming motor control beside the one inherited from recognition, answering the speed and velocity limits of Section 4.1.3 (Section 4.3.1). Second, let the model read the metric parameters and the decisions each action unit needs directly of one encoded frame (Section 4.3.2). Third, move the camera to see better, which answers the detail limit and requires a policy that acts correctly from any viewpoint (Section 4.3.3).

## 4.3.1 Add a fast streaming encoder for control beside the slow one for recognition

Feed a camera-rate encoder straight into the reactive policy head. One image encoder cannot supply detail and streaming rate at once (Section 4.1.3). We should learn from the two visual streams of Section 4.2.1 and give control its own encoder. The recognition encoder stays as it is; its tokens carry identity, layout, and task context for planning and action-unit selection. The control encoder runs at camera rate and feeds a few short-lived motion tokens per frame straight into the reactive policy head, which corrects the action without waiting for a new detail frame. Both streams enter one model, which learns when a decision needs detail and when the latest motion sufices.

Distill the fast encoder from video models, then train it on control losses. Much of the needed dynamics already exists in video foundation models [18, 292], whose pixel-generating variants spend their capacity on appearance a controller never uses [380], while latent prediction does not [270]. Policies that use optical flow [421] or point tracks [406, 36, 65] show that explicit motion helps manipulation. The fast encoder can be distilled from these sources and then trained with the policy on control losses, predicting the next action and the metric quantities of Section 4.3.2, with no reconstruction term.

## 4.3.2 Read the numbers and the decisions an action unit needs of one encoded frame

Encourage the model to emit more of each unit’s parameters directly from vision. Code-as-policy systems send Cartesian pose targets, each a set of numbers read from the scene: a grasp point and pose, a slot angle, an insertion depth [239, 152, 356]. Today those numbers come from code the model writes at run time, thresholding, segmenting, or line fitting [217, 118], or from a separate grasp or keypoint detector [105, 172]. The path works, but each step costs a round of generation and execution, and the thresholds break on the next object. We propose a complementary path, the direct readout of Section 4.2.2: given the unit, the model selects the features that matter, the slot’s axis for orient or the graspable part for grasp, and emits their numbers in the body frame, as pointing and spatial VLMs do in 2D [440, 286, 63] and spatially trained VLAs begin to in 3D [324]. The control encoder of Section 4.3.1 should carry the same kind of numbers at camera rate: optical flow, time-to-contact, and the approach of the hand to a surface, the quantities of Section 4.2.2, so that the reactive head reads a metric state instead of inferring it from pixels. The code path supplies the training data, as Section 6.3.4 describes, so measurement migrates into weights while the code remains as the test.

Answer the unit’s decisions with parallel typed experts over one encoded frame. A unit also needs decisions from the image, such as whether the handle is free. We propose typed experts inside the model, routed as in a mixture of experts [107]: the frame is encoded once, each question goes to the expert of its type, and the experts run in parallel, the channels of Section 4.2.2. Each returns a probability, a choice, or a score that downstream code reads directly (Sections 6.3.1 and 6.3.7), about ten times faster than generating text [298].

## 4.3.3 Move the camera to see better, and train across viewpoints

Move the camera to see better. We propose that the policy treat its wrist or head camera as an efector as well as a sensor and learn when to move it [25, 10, 71], as the eye is moved in Section 4.2.3. Moving closer puts more pixels on the target, and a new angle removes an occlusion that no processing could see through [26]: when a grasp is ambiguous, the policy should choose a pose that reveals the handle or the occluded side before committing the arm (Figure 15). Ambiguity has a measure: a grasp question of Section 4.3.2 answered with a probability near one half calls for a new view before any further reasoning. Moving closer also answers the detail limit (Section 4.1.3) with no change to the encoder.

![](images/166eda07a3d30b22b31fd4c71ee98d18a4e0ad1df1208cc911e136eeee83021b.jpg)  
Figure 15: A second viewpoint reveals what the head camera cannot see. The mug body partly hides its far-side handle from the head camera. Moving the wrist camera of the free hand to a side viewpoint reveals more of the handle before grasping. Teal arrows mark viewing directions; the curved gray arrow marks the wrist motion.

Give the model its camera pose, and train it to use it. We propose that every visual input arrive with its intrinsics and extrinsics, and that the model’s outputs be expressed in the body frame (Section 4.2.4). For the code path, the model should read the points the code needs as 3D coordinates in the body frame, using the calibration in its context; the code then runs unchanged from any view, and points gathered from several views fall into one scene (Section 4.3.2). For direct action, the policy should condition on the same calibration [183], so that a new pose is a new measurement of the same scene rather than a new task; views registered in the body frame become one token set the model attends over, which is the fusion. We propose recording each episode from several calibrated cameras [418], moving the wrist camera during episodes, and including tasks that no single view can solve, so that the model is forced to combine views.

# 5 Is the cerebellum the brain’s world model?

![](images/f05ad2f05733fd6d2aeb36bcc30134e6beee9cb895fd6a4859f5221b4a6d7a5f.jpg)  
Figure 16: Having tossed the ball herself, she knows from the throw where and when it will land. A copy of the tossing command is enough to forecast the ball’s arrival, so the right hand is already in place while the ball is still in the air. The faint circles show predicted positions of the same ball.

“World model” is robotics’ name for a model that predicts how the future unfolds, most visibly as generated video. The brain has its predictor too: the cerebellum takes a copy of each outgoing motor command and predicts, in a compact sensory code, what its owner will feel and see (Figure 16) [412, 269, 201]. Without it a person still acts, but coordination degrades badly [29]. With it, the prediction arrives within tens of milliseconds and recalibrates itself from its own error. The chapter compares the predictor in robotics (Section 5.1) with the one in the brain (Section 5.2), and proposes how a predictor beside the policy could let a robot adapt as quickly to its world (Section 5.3).

## 5.1 Robotics

## 5.1.1 World Action Models train the policy under a world-prediction loss

World models learn environment dynamics at scale from internet video [52, 292, 18]. Robotics has traditionally trained the world model apart from the policy [143, 146]: the policy was a separate small network reading the world model’s latent, so the prediction loss never shaped the policy’s own weights. World Action Models (WAMs) instead train one model on the joint distribution over future states and actions, with the action as one output [399]. The policy then inherits the prediction objective as a dense signal on the shared backbone, which encourages it to encode task-relevant temporal structure [238].

![](images/bf03b94d585e484bc455dfd47b6c0f4baf08fce90c28ff7acf6731c7924b8d31.jpg)  
Figure 17: Joint World Action Models let the prediction loss train the action head, while cascaded ones only pass predicted frames forward. In the cascaded design a video generator synthesizes predicted frames and a separate inverse-dynamics module reads actions of them. In the joint design a shared backbone trained under one generative objective feeds a state head that generates future frames, which some variants skip at test time, and an action head that emits the executed chunk.

How prediction is coupled to action decides what reaches the policy. In the cascaded family (Figure 17), a generative model imagines how the scene will evolve and an inverse-dynamics model reads of the actions between predicted frames [96, 399]. The generator needs no action labels, but rollout errors reach the actions uncorrected and action quality never trains the generator. The field has therefore moved to joint designs, where one network outputs both future frames and actions [399]. The discrete route predicts image, text, and action tokens in one autoregressive sequence, so an early action error propagates [58]. The continuous route denoises video and actions together, from one joint latent with decoupled heads [236] or with a separate noise level per modality [454], so the same network trains on video without action labels; latent actions inferred from optical flow [38] and co-training on action-free video, human hands included [431], add supervision from any embodiment.

The video branch that supplies the prior is also the source of the open problems. Latency comes from imagining before acting rather than from model size. Backbones span 0.4–14B parameters [238, 431], overlapping mainstream VLAs, yet WAMs that imagine before acting take 0.3–1.2 s per chunk, and up to 5 s for the largest video model, against 60–200 ms for VLAs; skipping generation at test time brings Fast-WAM to 190 ms, on par with mid-size VLAs [449, 439]. Evaluation is unsettled: emerging benchmarks score visual fidelity, physical commonsense, and action plausibility [399], but across the leading video generators physical understanding is poor and uncorrelated with visual realism, so a model can generate well while predicting dynamics that would fail on a robot [277]. The same branch also bounds what a WAM can predict: existing WAMs predict almost exclusively in the visual modality, so contact-rich settings where the relevant state is occluded or only tactile remain underexplored [376].

## 5.1.2 Test-time adaptation stores recent experience in a latent, in temporary weights, or in the context

Test-time adaptation changes how a policy behaves during deployment without a new training run, because the robot’s condition, the objects it handles, and the environment all drift. Three stores are in use, none specific to world models, and they difer in capacity and in cost.

A latent compresses recent experience into one vector. A network summarizes recent experience into a vector that conditions the policy: Rapid Motor Adaptation regresses an “extrinsics” latent from proprioceptive history in one forward pass [216, 362, 19]. The latent is instantaneous and sample-eficient but bounded by what it can represent.

Temporary weights hold more than a latent can. Parameters are written by gradient descent during deployment and discarded afterwards. RoboTTT updates test-time-training layers inside the policy on 8,000 timesteps of history [184, 366]; AdaJEPA takes one self-supervised step on a world model’s final layers per replanning cycle [402]; test-time training of the VLA follows the same pattern [182, 243]. Updates stay few and small, and nothing is written back, which separates this from continual learning

The context window holds the raw history. The raw history itself conditions the policy: a sequence model reads recent observations, actions, and outcomes and changes its behavior within the episode. People can supply that history as a few demonstrations [117, 390], or the robot can gather it, so that a legged robot walks better after its early falls [248, 224]. It is the most general store, but attention cost grows with the window and the experience is lost once it scrolls out.

## 5.2 Biology

A WAM needs hundreds of milliseconds to seconds per prediction (Section 5.1.1); the cerebellum answers in tens of milliseconds. From the state of the body and the eference copy of the outgoing command, it predicts what that command will produce: grip force tracks the load a lifting arm creates, and patients with cerebellar atrophy show impaired coupling and excess grip force [291]. The cerebellum holds about four fifths of the brain’s neurons [154]. What that investment buys is not settled; the forward-model account reads it as prediction [413, 181], and this survey adopts that reading.

## 5.2.1 A three-synapse, very wide circuit predicts before the senses report

![](images/b02aa003f536cda578499b58e5163a6cd94719fcd9366fab375d9e443e647009.jpg)  
Figure 18: The cerebellar circuit is three synapses deep, wide in the middle, and local along its length. The highlighted paths trace one local circuit: parallel fibers run only a few millimeters [315], and about forty Purkinje cells converge on one nuclear neuron [309], so coupling between body parts is local by construction. The inferior olive (amber) returns the prediction error over one climbing fiber per Purkinje cell, which adjusts the parallel-fiber synapses that were active when the faulty prediction was made. Counts are rough human figures [12, 285, 202].

The cerebellar circuit is a feedforward network only three synapses deep, narrow at both ends and enormously wide in the middle (Figure 18). Input arrives over the mossy fibers, which carry the eference copy and the current state of the body. The middle layer expands this input by about two hundred granule cells per mossy fiber [9, 202], into 50–100 billion granule cells, more than half of all neurons in the brain [12, 154], each sampling only 4–5 mossy fibers, so similar inputs are spread onto nearly disjoint sets of cells [260, 9]. The output layer contracts the code again: granule axons, the parallelfibers, cross the dendrites of 15–30 million Purkinje cells [284, 12], each of which sums ${ \sim } 2 \times 1 0 ^ { 5 }$ of them near-linearly [285, 394]. On the forward-model account, the Purkinje population discharge is the prediction: the anticipated sensory consequence of the command just issued. Width costs nothing in time; only depth does. A whisker touch in mice changes Purkinje output within 6–8 ms [51], and the nuclear output drives motor cortex through a fast loop [122], well inside the 50–100 ms that sensory feedback needs to alter a command through cortex [321].

## 5.2.2 Error signals recalibrate the cerebellum locally and fast

The climbing fiber delivers the error to the synapses that made the prediction. The brainstem reports the outcome of each command to the inferior olive within tens of milliseconds of the outcome. In the modern reading of the Marr–Albus–Ito theory, inhibition from the nuclei subtracts the prediction, so the olive fires mainly for the miss [265, 253]. The miss returns over the climbing fibers (Figure 18), one per Purkinje cell, each wrapping its target’s dendrites and firing it as one complex spike [295, 358]. The spike drives one well-documented learning rule: the parallel-fiber synapses that produced the faulty prediction, those active one loop delay earlier (a delay that difers by region and is about 120 ms in the flocculus [368]), are weakened in proportion to the miss [260, 9, 180], or strengthened, depending on the microzone [86], so the corrected prediction appears when the same input recurs.

Wiring gives the error its direction, and the expansion keeps the correction local. Purkinje cells are grouped into microzones, strips of about a thousand cells that share one target in the deep nuclei and so form one output channel [16]. In the best-studied zones, each microzone moves the body along one axis and its climbing fibers report error along that same axis, so a proportional change reduces it; the alignment is laid down largely by developmental wiring [16, 363]. The granule-layer expansion of Section 5.2.1 then keeps the correction local, whether through sparse activity, as few as 1–2% of cells active in any millisecond in the classical theories [260, 202], or through expansion and decorrelation alone: the synapses active during one prediction overlap little with those of another, so adjusting them leaves most others intact.

One error produces a brief correction; repeated errors consolidate over days to weeks. A single pairing depresses Purkinje firing on the next trial; repeated pairings stabilize it, and the trace shifts to the nuclei over about a week [428, 355, 86]. Magnifying spectacles leave the vestibulo-ocular reflex short of the image, and retinal slip raises its gain within hours [179]; reversing prisms invert its direction only over weeks [128]. How cortex uses the recalibrated forecast is described in Section 6.2.3

## 5.2.3 The prediction amends the action in flight or returns to cortex

![](images/786bd1cf59c7ee9254fb24a10301a5779a84dd1fcb0d42d7974c985c9d8a66aa.jpg)  
Figure 19: The same nuclear prediction leaves by two routes. Ascending (left), it reaches cortex through the ventrolateral thalamus. Descending (right), it joins the command in flight at the brainstem and spinal cord. Both arrive within tens of milliseconds of the command, before sensory feedback could alter it through cortex.

The nuclear output prediction leaves by two routes (Figure 19): down to the brainstem, to amend the command in flight, and up to the cortex that issued it.

Descending, the prediction amends the action in flight. The nuclei project directly to the brainstem motor centers and the spinal cord [119, 336], so the prediction can alter a movement without passing through cortex. In a reach, nuclear activity scales with how hard the arm must brake, and driving it changes the reach in flight [31, 252]. The fingers are a partial exception: their muscles depend heavily on direct cortical projections, so much of the correction reaches them by the slower route through cortex [230, 330].

Ascending, the prediction returns to the cortex that issued the command. Through the ventrolateral thalamus the nuclei reach the motor and premotor areas in closed loops [204, 47]. This input drives motor cortex firing during a reach [337], and silencing the nuclei collapses preparatory activity [122, 60], which suggests cortex plans on where the limb will be. Somatosensory cortex attenuates the predicted touch, so self-touch feels weaker [44, 207].

The cerebellum also sets timing and reaches the reward system. A fixed-latency nuclear signal triggers a movement and sets its timing [84, 218]. The cerebellum is also wired into the reward system: its nuclei drive dopamine neurons [56], and its own circuit carries the expectation of reward alongside the sensory forecast [391, 214]. A prediction therefore comes with a value attached, and the forecast feeds the system that scores practice (Section 6.2.4).

## 5.3 Inspiration

The cerebellum predicts ahead of the senses and recalibrates from its own error. Our proposals follow in decreasing confidence. Run a self-labeling predictor beside the policy and let the policy consult it as a tool; once the predictor answers before the sensory report arrives, let the policy steer by the forecast in the loop. Make the predictor shallow, wide, and sparse, so that it is fast and an error-scaled step within the episode corrects only the forecast that was wrong. Tag every joint and sensor with a body-part embedding so the sensorimotor map transfers across embodiments.

## 5.3.1 Consult the predictor as a tool when it is slow

We propose a predictor that runs beside the policy, receives the state and a copy of each command, and forecasts what the sensors will report, so that the processing delay is covered. It is the easiest module to train: every executed action is labeled by its own outcome a moment later. Even when too slow for the control loop, as current WAMs are (Section 5.1.1), it makes the search for a correction cheap (Section 6.3.7): a candidate correction is scored as a forecast before the robot moves [256, 148]. If it predicts that the object will not lift under the planned grip force, the model raises the force first, and the gap between forecast and outcome calibrates its model of contact.

## 5.3.2 Run the predictor in the loop when it is fast

A predictor fast enough to return before the real observation arrives changes what the model acts on: it steers by the forecast, so the delay between perception and inference drops out of every correction [370, 1]. The main benefit is that prediction and policy stay separate. A wrong outcome scores each against its own target, so the world’s dynamics are learned by the predictor alone and the policy’s weights stay intact [262, 212]. This suits a world that changes faster than a strategy should: the way the task is handled stays fixed while the model of the physics underneath it adapts within the episode. No model today infers fast enough for this; it is a direction for the future, open only to a predictor built for speed (Section 5.3.3).

## 5.3.3 Make the predictor shallow, wide, and sparse, so it is fast and adapts at test time

We consider a predictor that is shallow, wide, and sparse. Shallow buys speed, since latency grows with depth while width runs in parallel; current world models are deep, and depth sets their latency. Wide keeps capacity: expanding the input into a code many times wider makes nonlinear relations linear, so a simple readout sufices [79, 326], and the expansion decorrelates overlapping inputs [57], as the cerebellar granule layer is thought to do [260, 9]. Sparse makes test-time adaptation safe: in a dense model a gradient step moves shared features and shifts every forecast built on them [262], which AdaJEPA mitigates by updating only the final layers [402].

Sparsity is where a mixture of experts enters, in the delegated sense of Section 6.3.5 rather than the load-balancing sense of today’s models, whose experts rarely map onto nameable functions [345, 107, 259]. Groups of units are delegated to particular functions, a payload regime or a body part, so that overlapping inputs select nearly disjoint groups. Test-time adaptation then touches only the units behind the wrong forecast: one step scaled by the error corrects the current regime

and leaves every forecast with a disjoint support unchanged, so a heavier payload never disturbs an earlier calibration.   
The exact design, how deep, how wide, and how functions are assigned to units, is to be determined.

## 5.3.4 Tag every joint and sensor with a body-part embedding so the map transfers

We should replace the arbitrary joint index in VLA, world model, and multimodal large language model (MLLM) backbones with a body-part embedding. Cross-embodiment training today flattens each robot’s state and action into one vector, padded to a common length [296, 294, 93], so an element’s index is all that identifies it, and the index says nothing about which body part it belongs to or which element of another robot it corresponds to. Instead, each joint, actuator, and sensory channel should carry a learned vector named by its role, thumb, index, wrist, and the like, so every token has semantic as well as positional meaning. This structure need not be learned from data: the robot’s kinematics fix which actuator moves which sensor, as the cerebellum’s developmental wiring does (Section 5.2.2).

Two gains follow. For cross-embodiment transfer, human and robot hands trained under shared finger names can pool their data; we expect grip-force statistics for a thumb to carry across morphologies, as morphology-aware tokens already transfer across bodies [343, 369]. For adaptation within one embodiment, the forecast error routes by body part: when an object is wider than expected, the error lands on the thumb and middle-finger channels and corrects only the unit delegated to them (Section 5.3.3). Per-embodiment stems approximate this without naming joints [398, 322]; the named tag makes the structure explicit inside one model.

# 6 Planning and command: how the brain divides what a VLA does in one network

![](images/18c39b82ee7415a0a4bb1e3fb3bbd5e4327a3903ac61a72e09d924ebd4b383dd.jpg)  
Figure 20: One model composes familiar primitives into a new solution. Reach, grasp, orient, and tool use, none of them learned for this task, combine to retrieve an out-of-reach key, and the same model turns the plan into movement.

An action policy is a model that takes in observations and acts on them. Robotics has three forms of it (Section 6.1): a VLA, one network that outputs actions; a slow planner that issues subgoals to a VLA; and, most recently, a multimodal large language model that writes the policy as code. Motor cortex serves the same function with a diferent structure: separate areas plan, choose, release, and verify each action, and practice trains them for life (Section 6.2). We propose one model that composes action units from primitives (Figure 20), turns each unit into motion, and keeps learning after deployment by moving each primitive from code to weights, from explicitly parameterized perception and action to parameters held implicitly in the weights (Section 6.3).

## 6.1 Robotics

In this subsection we cover the VLA’s architecture (Section 6.1.1) and its training stages (Section 6.1.2), the planner over an executor (Section 6.1.3), the multimodal model that writes the policy as code over tools (Section 6.1.4), and the state of continual learning and recursive self-improvement (Section 6.1.5).

## 6.1.1 A VLA is a backbone plus action head

A VLA is a vision-language model with actions attached. The backbone brings visual and language grounding from internet-scale pretraining, and behavior cloning on robot demonstrations adds the motor output [49, 42]. In every form the output is a chunk of future targets, typically 0.5–2 s at 10–50 Hz (Sections 2.1.3 and 2.1.4); the forms difer in how the chunk is produced (Figure 21).

The first form treats actions as text. Each action dimension is binned and written onto the vocabulary’s least-used tokens, and the backbone decodes the chunk one token at a time [49, 210]. Taken to its limit, this gives an omni model in which images, text, and actions share one token sequence and one set of weights [58, 403]. The costs follow from the same choice: binned high-rate actions form long, highly correlated token streams that train poorly, decoding one token at a time is slow, errors accumulate along the chunk, and with no separate head the whole backbone runs at the control rate [310, 58].

The second form attaches a separate action expert, a smaller transformer that attends to the backbone’s features and emits the chunk as continuous values by difusion or flow matching [70, 241]; � , CogACT, GR00T N1, and SmolVLA all take this form [42, 235, 40, 354]. The head is generative mainly because demonstrations are multimodal: the same scene admits several valid trajectories, and a regression head would average them where sampling keeps one [70, 450].

![](images/ed6ce3bb101ff272e4b9398d9868bfdf4a13c46a8f30686fc110a8cd5039dc90.jpg)  
Figure 21: Actions attach to a vision-language backbone in one of three ways. (a) The first generation bins each action dimension and decodes actions as text tokens, one at a time. (b) The current generation keeps the backbone and attaches a small action expert that attends to its features and denoises a noisy chunk into continuous actions by difusion or flow matching; proprioception enters the expert, as drawn, in $\pi _ { 0 }$ and GR00T N1, and the backbone in SmolVLA [42, 40, 354]. (c) An omni model has no head. Images, text, and actions are tokens in one sequence, so the same network predicts the next action and the next image. Purple: transformer with pretrained weights. Teal: images, text, proprioception, and the action expert. Gray: discrete action tokens and the noise the expert denoises.

## 6.1.2 Three training stages shape the VLA, and none writes deployment experience back

Pretraining imitates demonstrations, and demonstrations are scarce. Pretraining teaches the inherited visionlanguage backbone to act by behavior cloning on pooled multi-robot data, usually with a difusion or flow-matching loss [301, 159, 241]; the loss rewards only fidelity to the demonstrator. Such data are scarce: the largest pooled datasets hold about a million episodes [296, 6]. Generalization grows with the variety of scenes, objects, and tasks, while extra demonstrations of the same scene and object stop helping after about fifty [240, 349]. Human video is plentiful but records no robot actions, which must be inferred as latents between frames or from hand pose [430, 163].

Supervised fine-tuning adapts the policy to one robot and keeps the imitation objective. Because pooled data mix embodiments, fine-tuning on tens to hundreds of target-robot demonstrations fixes the action space, camera placement, and control rate, through full updates or low-rank adapters [210, 42, 166]. Imitation copies every demonstration with equal weight, so a smaller, consistent set often beats a larger one that mixes operators’ strategies [32]. Recent systems add the policy’s own rollouts, labeled by quality so the model conditions on poor data instead of copying it [312, 401]. The objective still rewards fidelity alone, so a policy that drifts of the demonstrated states compounds its errors: the data hold no recoveries [332].

Reinforcement learning is the only stage at which success enters the objective. A reward scores the states the policy actually reaches, so its own failures become training signal. The experience can come from logged data reweighted by a learned value [233, 215], expert interventions at the moment of failure [203, 254], a learned simulator [234, 455], or the real robot [446, 68, 229]. At frontier scale the value verdict enters the context as a token [313], the critic trains the flow head directly [401], or a residual policy corrects a frozen VLA [15, 416]. In each case the policy surpasses its demonstrators within hours of real interaction on a single task [254, 68, 401]. Yet this fixes only the failures that were collected; deployment brings new payloads, shifted dynamics, visual corruption, and wear, met with frozen weights.

## 6.1.3 A planner decomposes the task into subgoals for a fast executor

A policy fast enough for the control loop is small, and a small model plans poorly. A slow planner supplies what it lacks: in the System 1 and System 2 split [195, 81, 452], a 1–7B-parameter VLM at 7–10 Hz interprets the task and a small transformer at 120–200 Hz acts on it [109, 40]. What passes between them trades legibility for bandwidth. Text is the most legible: the planner issues atomic commands [348] or writes the subtask before its own action chunk [314], which generalizes better and can be corrected [33, 444], but carries little pose and no contact or timing. A subgoal image adds the goal layout, where each object and the hand should end up and what should be touching what, though still no force or timing: a generative model imagines the next scene and conditions the executor on it [41, 312]. A latent embedding carries the most but is opaque and trained jointly with its executor [346, 109, 64].

## 6.1.4 A multimodal large language model writes the policy as code

The third design is a general multimodal large language model with reasoning and coding ability, such as GPT-6 Astra [297]. It writes a program over a library of perception and motion routines, so the plan is explicit code [239, 171, 278]; Cartesian control is the default, and a VLA is one option, called when the task is one it has been trained on [448, 420]. Because its competence comes from reasoning and coding beyond robot data, it generalizes beyond any demonstration set: zero-shot it ranks first among public policies at about 22% success [447, 120]; handing familiar manipulation to a VLA roughly doubles that in some cases [120], and a harness lifts it further [245, 448, 211].

The limit is time and reactivity. Each call costs tens of seconds of thought, so by the time the arm moves, the plan describes a scene that has already changed [120]. And because that call is the only loop, nothing closes the gap in between: a slipping grasp waits for the next round of reasoning [120, 447].

## 6.1.5 Experience enters the model in context, as skills, or in the weights, and no loop yet runs without a human

Past experience reaches the model by three routes. A human can provide a demonstration to the context [117, 344, 312], or the robot can refer to its own history there [248, 266]; both help, but the context has a size limit, so what scrolls out is lost (Section 5.1.2). A coding agent can write what it learns into a playbook [211] or a library of verified routines, where a record of past outcomes lifts real-robot progress several-fold [448, 244, 360]. Rollouts and corrections can be post-trained into the weights ofline [312, 313] or streamed from a fleet and redeployed within hours [401, 5], and a small replay fraction prevents most forgetting [246, 167].

![](images/089389a60983f7fb22a3359c42f9db577736530b85385452592dfce6365715c7.jpg)  
Figure 22: A robot learns mostly before it leaves the factory and afterwards executes the learned policy; a human begins with a genome rather than pretrained weights and does most of its learning after birth, through play, practice, and observation.

Automating the loop is the open part. The three routes are automated unevenly: agents already judge their own outcomes and write skills back, while choosing what to attempt next, deciding when a weight update is warranted, and verifying that an update is a real gain still fall to people. A robot that does all of these itself, as coding agents do for software, exists only in pieces [7, 342]. The latest attempts show what works. Attributing each repair to the skill that failed and validating it before reuse keeps improvements adding up across tasks [407, 397]. Matched controls that separate retained experience from repeated attempts make a success curve credible, and an independent evaluator of each update is the bar for recursive gain [400, 453]; Section 6.3.4 proposes the loop.

## 6.2 Biology

Current evidence suggests that the cortical motor system issues action commands in four stages: a goal is decomposed into action units whose progress is monitored (Section 6.2.1), each unit is probably composed from primitives that motor cortex already holds (Section 6.2.2), the cerebellar forecast may amend each unit in flight and time the next (Section 6.2.3), and practice appears to convert the composition into a habit (Section 6.2.4).

## 6.2.1 Prefrontal cortex sets the goal, SMA holds the action units, and ACC keeps the place

Prefrontal cortex holds the goal: its neurons carry a rule across a delay, independent of the stimulus and of the movement [271, 393]. It orders goals in a hierarchy and holds a sequence with each rank in its own subspace [213, 417]. The goal enters the motor areas through the supplementary motor area (SMA), which runs a sequence from memory where premotor cortex runs it from the senses [282, 283].

SMA holds the action units and their order: a unit is represented as a whole, and each movement within it by the slot it fills [371, 352, 76]. People form such units without instruction as they learn a long sequence [334], and pre-SMA, the front of the area, launches each unit and switches away from one in progress [205, 178]. Anterior cingulate cortex (ACC) keeps the place: its neurons track how many units remain before reward, and a subgoal reached evokes a prediction error like a reward’s [350, 329]. Unlocking a door is such a sequence: reach, orient, insert, and rotate, each unit ending at a sensory event the next waits for (Figure 23).

![](images/4faad7cdb0c9e6c736f49211330264a5e6f61918ae8b3ebe02363d48ce773183.jpg)  
Figure 23: Unlocking a door is four action units in a fixed order, reach, orient, insert, and rotate, and each ends at a sensory event: the tip reaches the slot, the key aligns, it stops at the end, the lock turns.

## 6.2.2 Motor cortex holds sensor-tuned primitives, and a competition among premotor candidates chooses which to compose

Motor cortex is primary motor cortex (M1), the last cortical stage before the spinal cord, plus a set of premotor areas that feed it [364]; we focus on premotor cortex and SMA (Figure 24). Its repertoire is a set of primitives: stimulating premotor cortex or M1 in macaques for half a second evokes a whole action, the limb ending in the same posture wherever it started [138, 135]. Each primitive has a sensory input already tuned for it: reach, orientation, and grasp run on separate parieto-premotor circuits [121, 106, 280], and ventral premotor neurons fire at the sight of an object sized for the grip they command [279].

Which action to take is decided by competition among candidates. Dorsal premotor cortex can hold several potential reaches at once, each scaled by its likelihood, until one grows and suppresses the rest [74, 124], and the prefrontal rule tilts the competition toward the one that serves the goal [393, 75]. M1 activity correlates with the force, direction, and speed of the movement in progress [125], though population dynamics rather than parameter coding may be the better description [73].

![](images/2bd28ae5460d1fa37f129b731aa7e96953ed7c430ab6e9dee161f6462c6af9e6.jpg)

Figure 24: Motor cortex is M1, premotor cortex, and SMA, reciprocally connected, with three cortical inputs: prefrontal goals enter through SMA and premotor cortex, the parietal state enters through premotor cortex and, from S1 and area 5, M1 directly, and ACC reaches SMA and M1 through the cingulate motor areas [364]. M1 carries most of the corticospinal tract, and SMA, premotor, and cingulate areas each add a direct projection that bypasses it [100]. Motor cortex also drives the brainstem pathways and runs loops through the basal ganglia and cerebellum [230]; reflex aferents return to the cord and touch and proprioception return to parietal cortex. Only the principal pathways are drawn.

## 6.2.3 The cerebellum’s forecast amends the current unit and times the next

The cerebellum forecasts the consequence of each command and returns the forecast by two routes, descending to amend the command in flight and ascending to the cortex that issued it (Section 5.2.3). For the action unit, the two routes do diferent work.

Descending, the forecast corrects the unit in progress without ending it. A small error is absorbed implicitly, without awareness [372, 208]; a large one also prompts an explicit change of strategy at the boundary [263].

Ascending, the forecast tells cortex where the current unit will end, and its gap from the goal is what cortex plans on: if the unit will fall short, cortex adjusts the next unit or replans the strategy before the shortfall is sensed, which is the forecast’s main service to the planner. The next unit forms in an output-null subspace of premotor cortex and M1 where it cannot reach the muscles [199, 72], and cerebellar input sustains it and times its release (Section 5.2.3). Release runs through a midbrain–thalamus–cortex trigger that the basal ganglia otherwise hold shut [175, 274]; whether the cerebellar signal also reaches that gate is open (Figure 25).

![](images/d369a2c1a74e3b35a1f42f918905a69f29172296891cd3c227b7052216d989c3.jpg)  
Figure 25: One action unit runs as a loop; the arrows show the flow of control rather than anatomical pathways. Purple marks the four stages, run clockwise, and teal what each stage reads. Solid teal arrows have direct support: the cerebellar forecast amends the unit in flight [31, 252] and sustains and times the cortical preparation of the next [122, 60]. Dashed arrows are hypotheses: whether the forecast also times the basal ganglia release is untested, and pre-SMA, which launches units and switches between them [205, 178], is proposed here to do so on the ACC verdict. Assigning one area to each stage is a simplification; each area takes part in more than one.

## 6.2.4 Practice turns the composition into a habit, selected by dopamine and consolidated ofline

A new skill begins as a composition of primitives the repertoire already holds, acquired within a session where a pattern outside the repertoire needs days [157, 333, 293]. Practice turns the composition into a habit that runs largely below cortex. Striatal bursts come to bracket the whole sequence at its start and end [185, 134], and the basal ganglia then set its detailed kinematics [91]. In rats, M1 is required to learn a lever-press skill but not to execute it once learned, unless the sequence must be reordered on the fly [200, 275]. In one account, control shifts from cerebellar to striatal circuit along the way [95], and the co-contraction that stifened the limb while its model was wrong is released [374, 353].

Dopamine decides which synapses practice strengthens. Co-activity marks a corticostriatal synapse as a candidate, an eligibility trace [126], and a dopamine burst or dip, the reward prediction error, supplies the verdict [340, 116]. A spine enlarges only if dopamine arrives within about two seconds of its activation [423], and corticostriatal synapses strengthen over the days of training [433, 311]. Practice concentrates where motor variability rises [90].

Ofline replay consolidates what dopamine chose. New spines form in M1 within hours of practice, and retention tracks how many survive [422, 426]. A finger sequence runs about a fifth faster the next morning without further practice, and less so after an equal interval awake [392, 219], though how much of this gain is specific to sleep is disputed [302, 142]. During slow-wave sleep the ensembles active in the day’s task reactivate, and the reactivation predicts the next day’s gain and selects the synapses kept [327, 141, 427].

## 6.3 Inspiration

The proposals of this chapter combine into one agent. For a new task, a single multimodal model composes a solution from existing primitives, verifying each at its boundary (Section 6.3.1), and runs each by Cartesian control or a reactive policy head (Section 6.3.3); each primitive is a template the model parameterizes per instance (Section 6.3.2). The solution then matures through two agents: a practice agent rehearses its weakest unit (Section 6.3.6), guided by judging which direction moves an attempt closer to success (Section 6.3.7), and a consolidation agent prunes the strategy and graduates each primitive from explicit code into typed functions the model answers, then into the weights (Section 6.3.4). Beside the model, a predictor forecasts the consequence of each action (Sections 5.3.1 and 5.3.2).

## 6.3.1 Let the model compose action units from reusable primitives and verify each at its boundary

Splitting a task into steps is largely solved: GPT-6 Astra already breaks opening an unfamiliar lock into orient, insert, and rotate [297, 152]. What is missing is a library of reusable primitives, pick, place, orient, slide, insert, stored as skills or weights and shared across instances [239, 356, 118]. A new skill is then a code plan over them, as opening a lock is a plan over reach, orient, insert, and rotate (Figure 23), and where no primitive fits, the model writes a new one as code, since large models generalize to programs they have not seen [239, 356]. Each primitive holds two things. The first is a verifiable goal: a test the robot runs on its own sensors at the unit’s boundary, and the next unit starts only once that test has fired [395, 420, 62]. The second is an action plan, code or a policy head that pursues the goal (Section 6.3.3), written as a template whose parameters the model fills in for each instance (Section 6.3.2). When a test fails, the model must reason to the cause, a slot too narrow, a second object dragging the part back, and revise the plan or adjust the force accordingly. Current multimodal models rarely manage this diagnosis, least of all where forces interact out of the camera’s view [287, 140, 237].

## 6.3.2 Treat each primitive as a parameterized template

The plan of a primitive is generic; what changes from one instance to the next is its parameters: the grasp point and approach direction, the slot’s angle, the insertion depth, the force at which to stop. A correct plan with a wrong angle fails as surely as a wrong plan, so supplying precise parameters is where multimodal capability matters most (Section 4.3.2). Where the parameters come from is what changes as a primitive matures (Figure 26). At first the model writes code that measures them: segment the slot, fit a line, read of the angle. As the model sees more instances, it reads parameters of the image directly, one at a time, and the code shrinks to the measurements it cannot yet make reliably; the handover is gradual, with the code’s answer serving as the check on the model’s until the two agree often enough that the code is dropped. In the end the parameters are never named at all: the weights of the action head read them implicitly from the scene. The consolidation agent of Section 6.3.4 advances each primitive along this path.

![](images/6584090a401df044c9cabc31580f8ebc0531a29c013b964bb8727cd08a9d8f65.jpg)  
Figure 26: Consolidation advances a primitive from a script over many functions to an end-to-end policy. (a) At first the model writes a perception pipeline that computes the slot’s pose and the key’s geometry in the hand before calling the orient template, which runs by Cartesian control. (b) Next the model outputs both values directly from the images and supplies them to the same template, so the explicit measuring code is no longer needed. (c) In the end the template lives in the orient experts of the model’s mixture-of-experts trunk: the action head of the same model reads their latent, no number is passed, and the action stream reacts at the control rate.

## 6.3.3 Let the model run each unit by Cartesian control or by a reactive policy head

A primitive’s action plan takes one of two forms. The direct form is code: the model writes Cartesian pose targets, interpolates them at the control rate, and leaves contact to the compliance layer of Section 2.3. GPT-6 Astra does this today and succeeds where a task consists of reaches and placements [120, 152, 356]. Between one boundary test and the next, though, the trajectory is blind: a cable that snags is noticed only when the unit ends, and code holds no prior about how objects behave.

The proposed form is a lightweight policy head that runs the plan against real-time feedback at the control rate, its weights holding that prior [42, 40, 64]. The two forms compose: code sets a unit’s milestone poses while the head generates the trajectory between them and reacts to contact [448, 420]. The head reads its instruction as a latent from the same model, already carrying pose, contact, and timing, so it needs no second language backbone [346, 109]. Reacting this fast needs an equally fast vision encoder [69, 365], the streaming encoder of Section 4.3.1.

## 6.3.4 Add a consolidation agent that graduates proven primitives from code into weights

Between sessions, an ofline agent reviews the records of the boundary tests, answered by the typed experts of Section 4.3.2 that we call the judge, and keeps what paid of, as consolidation does in Section 6.2.4. It writes primitives, task sequences, grasp records, and failure explanations, and moves each proven primitive along the path of Figure 26: a cleaned script, then typed functions the model answers from the image, and in the end a policy with no code in the loop.

Consolidation first cleans the script, keeping the shortest sequence that is grounded and explained. The run-time code plan (Section 6.3.1) carries whatever happened to work alongside what made it work. The agent replays it against the stored variations of the scene and keeps the shortest sequence that still passes; a retry loop survives only where the replay fails without it [420, 254]. Every kept unit states its precondition and boundary test, and every value traces to a measurement or a stored record of the object: the trial-found ofset of Figure 27 stays only once it is explained by the measured slot and key. The script is then stored, keyed by goal and scene, for retrieval into context [396, 244].

Each measurement then graduates from code into a typed function the model answers directly. The stored script still finds the slot’s angle with a program that thresholds the image and fits a line [217, 66]. Each passed run pairs a scene with the number that worked, and training on these pairs lets the model answer the same call from the image, keeping the function’s signature and dropping its body [440, 144, 211]. Code and model answer the same inputs until they agree often enough that the code serves only as the test.

In the end the policy head runs the primitive end to end, and only the test stays explicit. A practiced primitive runs with no code in the loop: the head receives “screw the cap onto the bottle” and reacts to streaming visual and tactile input from start to finish, as a VLA does today [42, 69, 64]. The angle once measured is carried implicitly in a shared latent, and a part that binds is answered without passing through a scalar [346, 435, 237]. The boundary test does not graduate: as a person notices a cap cross-threading, the judge keeps asking its typed question at each boundary, because that verdict is what the loop trains on.

## 6.3.5 Design the next architecture so that primitives are experts routed by the plan and share embeddings

The current model is slow, and coding is slow [239, 120, 452]. The round trip of writing out text and reading it back in between the model’s reasoning passes is wasteful. If everything passed in between were embeddings, turned into human language only when the robot needs our feedback, it would be much faster [346, 109]. Many animals do not speak a language and accomplish hard action tasks [89, 387].

The logic that today lives in generated code, which unit runs next and what it receives, can instead be carried by routing between experts. An expert here is a small part of the large model that takes in an embedding and returns an embedding; the notion borrows from the mixture of experts, where today’s mixtures route tokens between experts for load balance [345, 107]. The diference is what an expert stands for. Here each primitive persists as a group of experts that behaves as a function call whose input and output are embeddings rather than numbers, and all groups read one shared encoding of the scene (Section 4.2.2). Orient activates the experts that align a part, insert those that regulate contact force, and a hand-of between primitives is a change of routing in place of a line of code [259, 435, 24]. More experts at fixed active parameters add room for new primitives at no cost per step.

The benefit is not only speed. Concepts no longer need to be re-encoded and decoded at every hand-of, which wastes much of today’s compute and time [452, 346]. An expert can also be trained from a small number of examples, since on well-trained input and output embeddings it is computationally on the level of a small multilayer perceptron [436, 92]. The routing itself needs learning: each expert would carry an address that is encoded, and the model learns the mapping from the plan to that address [345, 259]. The challenge is inspection. Once the output is no longer human language, it is hard for a person to inspect what training did, and a readout that decodes a latent to a number on demand stays for the judge. This is only our proposal, drawn with wide imagination, and we look forward to creative ways of developing the next generation of models for robotics.

## 6.3.6 Add a practice agent that rehearses weak skills and tests what consolidation wrote

We propose an autonomous practice agent that picks the most valuable tasks, rehearses their weakest unit, and tests each record consolidation wrote by the trial that would refute it (Figure 27).

![](images/2e2915d91277830dbb540bcbbd0cc8a18e3023a64c60f00a66975b4db8a0d7e2.jpg)  
Figure 27: The practice agent finds the error-prone part across attempts, corrects it during inference, and hands the corrected record to consolidation. (a) Over seven episodes of opening a lock, insert passes rarely, and the blame falls on orient’s angle, which is solved with an arbitrary ofset after several model calls and fails most first attempts; it is unfamiliar and error prone and is practiced first. (b) During practice the direction to move after each attempt comes from the graded goal test, the forecast against the outcome, or the model’s reasoning; the interval halves until the goal test fires, and the angle that passed is stored. (c) Consolidation then improves the coding strategy that finds the keypoints, trains the model to output the values directly, and trains the orient experts of the trunk, so the orient primitive becomes familiar.

It picks the most valuable tasks and practices their weakest unit. A task is valuable when its reward is high and the gap between the robot’s current success and the success it could reach is large: a task already solved earns nothing more, and one nobody rewards is not worth the practice. Within the chosen task, the model reasons over the records to decide which unit is to blame, since a failure that surfaces at one unit’s boundary may have been caused by the unit before it; the blamed unit is the weakest, and practice goes there first. Three signs mark it: it fails most often, it demands the most deliberation before it passes, extra model calls and search for a parameter that should have been read of directly, and its parameters vary most from one attempt to the next. All three are signs of inexperience: an experienced unit chooses the same primitive and parameters across attempts in similar states and reaches the goal without extra search or reasoning, while an inexperienced one has few attempts behind it and is still being composed [414, 89], and inconsistency between successive action predictions within a rollout predicts failure at run time [4, 341].

It treats each record as a hypothesis and runs the trial that would refute it. What consolidation wrote is a set of claims that can be wrong: that a failure was due to the pose, that pouring does not depend on the cup’s height. The practice agent schedules the one trial that tests the claim with everything else held, and a claim that fails is rewritten with the new evidence. Practice stops when the outcome variance inside it has fallen to the task’s tolerance.

## 6.3.7 Mastering a new skill rests on judging which direction moves it closer to success

A precise pose or force is rarely found on the first attempt, and it need not be: recursive self-improvement requires only that a direction of improvement be readable from a failed attempt, since whatever can be corrected can be improved without a human in the loop [342, 289, 254].

Correction rests on one judgment: which of two outcomes lies closer to the goal. Where the gap is near linear in the adjustment, as a distance is in a pose, the model reads the correction of the gap, the plug stopped short so advance [33, 444], and climbs that gradient until the unit’s test fires [254, 384]. Where it is not, as in the body posture for a badminton stroke, no single joint tells which direction leads to a better outcome, so the agent samples more widely to map the landscape before choosing a direction [89, 90].

Prediction is essential to the search, because it supplies the link between a change in the action and a change in the outcome. Forecasting the consequence of a candidate adjustment makes the gap between forecast and goal a near-linear function of the correction, so the direction to move is read of the forecast (Section 5.3.1) [256, 147]. The final corrected trajectory is handed to the consolidation agent (Section 6.3.4).

## 7 Conclusion

How the next robot model should be built is an open debate: a single vision-language-action policy (VLA) or world action model (WAM) [42, 312], code as policy [297, 448], or a slow system 2 paired with a fast system 1 [109, 452]. Whichever wins, humans remain far ahead of robots: we solve novel tasks, handle objects we have never held, react fast to a moving world, and keep improving for life, while policies trained on a million pooled trajectories do none of these reliably (Section 1). The nervous system is a working design that already does all of this, so this survey asked what robotics should borrow from it.

The two architectures difer in where the work is done. The robot stack is a model plus servo control. The model is a VLA or WAM, often with a planner or coding agent above it, that reasons or predicts at hundreds of milliseconds to seconds per step; it takes a language instruction and camera frames and emits target positions (Sections 6.1.1, 6.1.3 and 6.1.4). Servo control tracks those positions with fixed gains. The nervous system has the same two levels, but each is richer. The upper level is a set of cooperating modules: cortex, thalamus, basal ganglia, and cerebellum, which divide goal decomposition, motion planning, gating, reward, and prediction among them. The lower level receives motor neuron firing rather than positions; muscles turn it into force, and opposing muscle pairs set stifness as well as torque, so the body is compliant by construction. Table 3 sets the two side by side, stage by stage, with what we propose.

<table><tr><td>Stage</td><td>Current robot stack</td><td>Nervous system</td><td>Proposal</td></tr><tr><td>Action (Section 2)</td><td>A position target goes into fixed servo gains, and the same gains serve every contact.</td><td>The command is effort per muscle, and opposing pairs set torque and stiffness separately.</td><td>Emit position and stiffness together. Realize the stiffness beneath the policy by impedance or admittance, and let touch set where to yield and how hard to press.</td></tr><tr><td>Touch (Section 3)</td><td>Sensors cover only parts of the hand, and the signal differs per sensor family.</td><td>Four afferent classes cover the whole skin at high density and precision, measure both normal and shear force, and a grip reflex corrects slips.</td><td>Treat touch as a first-class input, and use it for compliance control.</td></tr><tr><td>Vision (Section 4)</td><td>One recognition encoder reads one stale frame with no velocity, and encoding is slow.</td><td>A ventral stream handles recognition, and a separate dorsal stream carries fast metric signals for guiding the hand.</td><td>Build a dedicated streaming vision encoder for control.</td></tr><tr><td>Predictor (Section 5)</td><td>A video world model or WAM takes hundreds of milliseconds per step and adapts poorly at test time.</td><td>The cerebellum forecasts within tens of milliseconds, recalibrated with a few trials by its own error.</td><td>Train a self-labeling predictor beside the policy. Consult it as a tool.</td></tr><tr><td>Policy and learning (Section 6)</td><td>A VLA copies demonstrations, a slow planner sits over a fast executor, code as policy is slow and blind between calls, and weights are frozen at deployment.</td><td>The goal is split into action units checked at their boundaries. Primitives live in motor cortex and combine to solve new tasks, and practice is scored by dopamine and consolidated offline into new motor skills.</td><td>One model composes parameterized primitives in code, verifies each at its boundary, and runs them through a reactive head. A practice agent rehearses the weakest unit, and a consolidation agent graduates each skill from code into weights.</td></tr></table>

Table 3: Discrepancies between the current robot stack and the nervous system, stage by stage, and the proposal each chapter derives from the gap.

Inspired by the neuroscience, here is our proposal for the next generation of robot systems.

One model composes the new task from verified primitives. The model draws on a library of reusable primitives, reach, orient, insert, rotate, each a template with a verifiable goal: a test the robot runs on its own sensors at the unit’s boundary. It breaks the task into action units and parameterizes each for this instance (Sections 6.3.1 and 6.3.2). The parameters are what change between instances, the angle of this slot, the depth at which this key seats, and the model reads them of the encoded frame when it can and writes measuring code when it cannot. When a test passes, with touch or vision matching the expectation, the model moves to the next primitive. When a test fails, it reasons to the cause, the key is upside down, the lock is stif, this is the wrong key, and revises the plan or adjusts the force. Where no primitive fits, the model writes a new one as code, since large models generalize to programs they have not seen.

Compliance and touch let the world supply what the policy would otherwise compute. The insertion need not know the slot’s exact pose. The policy emits a position and a stifness, high along the insertion axis and low across it, and the stifness is realized beneath the policy, by impedance where the joints deliver torque and by admittance on position-controlled arms (Sections 2.3.1 and 2.3.2). Touch supplies the rest, entering the model as a force field, a normal and shear force per skin point, which any sensor family can be calibrated to (Section 3.3.1). On the key, its magnitude says when to stop closing, and a sudden change signals slip long before vision would.

A reactive policy head and a streaming encoder make the motion fast and continuous. Code is blind between tests, and a plan executed pose by pose stops at every milestone. So each unit runs through a lightweight policy head inside the same model: code sets the milestone poses and passes down a latent, and the head turns that latent into actions at the control rate, adapting on the fly to small changes (Section 6.3.3). Vision must keep up. The current encoder is built for recognition, so we propose a second one beside it that runs at camera rate and feeds the head a few short-lived tokens per frame carrying what a reaction needs: optical flow, motion direction, object speed (Section 4.3.1).

A predictor beside the model forecasts the consequence of each action and trains itself. The predictor receives a copy of every command and forecasts what the sensors will report, so a correction can start before the report arrives. Training is free: each executed action labels itself a moment later, and because the robot caused the outcome, the link it learns is causal (Section 5.3.1). How the policy uses it depends on speed. While the predictor is slow, the policy consults it as a tool and scores a candidate correction before moving. Once it answers before the sensory report, the policy steers by the forecast and the sensing delay drops out of every correction (Section 5.3.2). A shallow, wide, and sparse architecture makes it fast and lets it recalibrate within the episode: a single wide layer projects the input into a very high-dimensional code and processes it in parallel groups of units, each delegated to one payload regime or body part, so an error-scaled step at test time corrects only the forecast that was wrong (Section 5.3.3). The result is a robot that recalibrates to a stifer lock or a heavier key within a single attempt, without touching the policy that decided to turn it.

Perception moves from explicit code into typed functions, then into the weights. This is the proposal we rate highest. Large language models once handed arithmetic and programming to a calculator or an interpreter; today much of that computation runs inside the model. We propose the same path for perception. The slot’s pose, the key’s orientation, and how tightly the fingers should close start as quantities the model computes explicitly in code; with repetition they become typed functions the model calls; with enough practice they become weights that produce the action from the image directly. Facing an unfamiliar lock, the robot measures: it estimates the pose from the image, confirms it with the hand, and checks each step before the next. After consolidation it eyeballs: it glances at the lock and acts, as a person does with a familiar door. Eyeballing works because compliance and fast online correction absorb an estimate that is of by a few millimeters or degrees, so moderate precision is enough to finish the task. Section 6 gives the details.

The robot should practice and sense on its own initiative. We propose a practice agent that picks the most valuable task and, within it, the unit that fails most often, demands the most deliberation, and varies most between attempts, then rehearses it until its outcome variance falls within tolerance (Section 6.3.6). Rehearsal pays of only if the model can judge how far each outcome falls short of the goal, so we propose to strengthen that judgment: a model that rates each state can correct toward the goal during inference, and as long as each correction narrows the gap, the attempts converge (Section 6.3.7). The same initiative extends to what the robot believes. Each record consolidation wrote is a hypothesis, and the agent schedules the trial that would refute it; each unseen property the plan depends on is a hypothesis too, tested before the robot commits: a small rotation of the key tells whether it is seated, a press gauges the lock’s stifness, a change of viewpoint exposes the slot (Sections 3.3.3 and 4.3.3). A robot that takes the initiative and learns actively is the core of recursive self-improvement.

Limitations and outlook. This is a position paper. The neuroscience it draws on is in part hypothesis, still debated in its own field, and the analogy has limits of its own: robots copy weights, pool records, and run many bodies at once, so some of the pressures that shaped the nervous system may not bear on them. Which of the proposals survives contact with real robots is an empirical question, and we ofer each as a hypothesis to be tested. The experiments that decide between these designs are within reach, and the agent they point to is worth building.

## References

[1] Ignacio Abadía, Francisco Naveros, Eduardo Ros, Richard R. Carrillo, and Niceto R. Luque. A cerebellar-based solution to the nondeterministic time delay problem in robotic control. Science Robotics, 6(58), September 2021. ISSN 2470-9476. doi: 10.1126/scirobotics.abf2756. URL http://dx.doi.org/10.1126/scirobotics. abf2756.

[2] Victoria E. Abraira and David D. Ginty. The sensory neurons of touch. Neuron, 79(4):618–639, 2013. doi: 10.1016/j.neuron.2013.07.051.

[3] E. D. Adrian and Y. Zotterman. The impulses produced by sensory nerve-endings: Part II. The response of a single end-organ. The Journal ofPhysiology, 61(2):151–171, 1926. doi: 10.1113/jphysiol.1926.sp002281.

[4] Christopher Agia, Rohan Sinha, Jingyun Yang, Zi-ang Cao, Rika Antonova, Marco Pavone, and Jeannette Bohg. Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress. arXiv preprint arXiv:2410.04640, 2024.

[5] AgiBot. SOP: A scalable online post-training system for vision-language-action models. arXiv preprint arXiv:2601.03044, 2026.

[6] AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, Shu Jiang, Yuxin Jiang, Cheng Jing, Hongyang Li, Jialu Li, Chiming Liu, Yi Liu, Yuxiang Lu, Jianlan Luo, Ping Luo, Yao Mu, Yuehan Niu, Yixuan Pan, Jiangmiao Pang, Yu Qiao, Guanghui Ren, Cheng Ruan, Jiaqi Shan, Yongjian Shen, Chengshi Shi, Mingkang Shi, Modi Shi, Chonghao Sima, Jianheng Song, Huijie Wang, Wenhao Wang, Dafeng Wei, Chengen Xie, Guo Xu, Junchi Yan, Cunbiao Yang, Lei Yang, Shukai Yang, Maoqing Yao, Jia Zeng, Chi Zhang, Qinglin Zhang, Bin Zhao, Chengyue Zhao, Jiaqi Zhao, and Jianchao Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[7] Michael Ahn, Debidatta Dwibedi, Chelsea Finn, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Karol Hausman, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Sean Kirmani, Isabel Leal, Edward Lee, Sergey Levine, Yao Lu, Isabel Leal, Sharath Maddineni, Kanishka Rao, Dorsa Sadigh, Pannag Sanketi, Pierre Sermanet, Quan Vuong, Stefan Welker, Fei Xia, Ted Xiao, Peng Xu, Steve Xu, and Zhuo Xu. Autort: Embodied foundation models for large scale orchestration of robotic agents. arXiv preprint arXiv:2401.12963, 2024.

[8] Alin Albu-Schäfer, Sami Haddadin, Christian Ott, Andreas Stemmer, Thomas Wimböck, and Gerd Hirzinger. The DLR lightweight robot: Design and control concepts for robots in human environments. Industrial Robot: An International Journal, 34(5):376–385, 2007. doi: 10.1108/01439910710774386.

[9] James S. Albus. A theory of cerebellar function. Mathematical Biosciences, 10(1–2):25–61, 1971.

[10] John Aloimonos, Isaac Weiss, and Amit Bandyopadhyay. Active vision. International Journal of Computer Vision, 1(4):333–356, 1988. doi: 10.1007/BF00133571.

[11] Amir Amedi, Rafael Malach, Talma Hendler, Sharon Peled, and Ehud Zohary. Visuo-haptic object-related activation in the ventral visual pathway. Nature Neuroscience, 4(3):324–330, 2001. doi: 10.1038/85201.

[12] Birgitte Bo Andersen, Lise Korbo, and Bente Pakkenberg. A quantitative study of the human cerebellum with unbiased stereological techniques. Journal of Comparative Neurology, 326(4):549–560, 1992. doi: 10.1002/cne.903260405.

[13] Richard A. Andersen and Christopher A. Buneo. Intentional maps in posterior parietal cortex. Annual Review of Neuroscience, 25:189–220, 2002.

[14] Richard A. Andersen, Greg K. Essick, and Ralph M. Siegel. Encoding of spatial location by posterior parietal neurons. Science, 230:456–458, 1985.

[15] Lars Ankile, Anthony Simeonov, Idan Shenfeld, Marcel Torne, and Pulkit Agrawal. From imitation to refinement - residual RL for precise assembly. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1–8. IEEE, May 2025. doi: 10.1109/icra55743.2025.11127442. URL http: //dx.doi.org/10.1109/icra55743.2025.11127442.

[16] Richard Apps and Martin Garwicz. Anatomical and physiological foundations of cerebellar information processing. Nature Reviews Neuroscience, 6:297–311, 2005. doi: 10.1038/nrn1646.

[17] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. I-JEPA.

[18] Mahmoud Assran et al. V-JEPA 2: Self-supervised video models enable understanding, prediction, and planning. Technical report, arXiv, 2025.

[19] Karl Johan Åström and Björn Wittenmark. Adaptive Control. Addison-Wesley, 2nd edition, 1995.

[20] David Attwell and Simon B. Laughlin. An energy budget for signaling in the grey matter of the brain. Journal of Cerebral Blood Flow & Metabolism, 21(10):1133–1145, 2001. doi: 10.1097/00004647-200110000-00001.

[21] A.-S. Augurelle, A. M. Smith, T. Lejeune, and J.-L. Thonnard. Importance of cutaneous feedback in maintaining a secure grip during manipulation of hand-held objects. Journal of Neurophysiology, 89(2):665–671, 2003. doi: 10.1152/jn.00249.2002.

[22] Marie Avillac, Sophie Denève, Etienne Olivier, Alexandre Pouget, and Jean-René Duhamel. Reference frames for representing visual and tactile locations in parietal cortex. Nature Neuroscience, 8(7):941–949, 2005. doi: 10.1038/nn1480.

[23] Shuai Bai, Keqin Chen, Xuejing Liu, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[24] Shuanghao Bai, Meng Li, Xinyuan Lv, Jiawei Wang, Xinhua Wang, Fei Liao, Chengkai Hou, Langzhe Gu, Wanqi Zhou, Kun Wu, Ziluo Ding, Zhiyuan Xu, Lei Sun, Shanghang Zhang, Zhengping Che, Jian Tang, and Badong Chen. HEX: Humanoid-aligned experts for cross-embodiment whole-body manipulation. arXiv preprint arXiv:2604.07993, 2026.

[25] Ruzena Bajcsy. Active perception. Proceedings ofthe IEEE, 76(8):966–1005, 1988. doi: 10.1109/5.5968.

[26] Ruzena Bajcsy, Yiannis Aloimonos, and John K. Tsotsos. Revisiting active perception. Autonomous Robots, 42 (2):177–196, 2018. doi: 10.1007/s10514-017-9615-3.

[27] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEiT: BERT pre-training of image transformers. In International Conference on Learning Representations (ICLR), 2022.

[28] Adrien Bardes, Quentin Garrido, Jean Ponce, et al. Revisiting feature prediction for learning visual representations from video (V-JEPA). arXiv preprint arXiv:2404.08471, 2024.

[29] Amy J. Bastian. Learning to predict the future: The cerebellum adapts feedforward movement control. Current Opinion in Neurobiology, 16(6):645–649, 2006. doi: 10.1016/j.conb.2006.08.016.

[30] Aaron P. Batista, Christopher A. Buneo, Lawrence H. Snyder, and Richard A. Andersen. Reach plans in eye-centered coordinates. Science, 285(5425):257–260, 1999.

[31] Matthew I. Becker and Abigail L. Person. Cerebellar control of reach kinematics for endpoint precision. Neuron, 103(2):335–348, 2019. doi: 10.1016/j.neuron.2019.05.007.

[32] Suneel Belkhale, Yuchen Cui, and Dorsa Sadigh. Data quality in imitation learning. In Advances in Neural Information Processing Systems 36, NeurIPS 2023, page 80375–80395. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023. doi: 10.52202/075280-3524. URL http://dx.doi.org/10.52202/ 075280-3524.

[33] Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. RT-H: Action hierarchies using language. In Robotics: Science and Systems, 2024.

[34] Pierre-Michel Bernier and Scott T. Grafton. Human posterior parietal cortex flexibly determines reference frames for reaching based on sensory context. Neuron, 68(4):776–788, 2010. doi: 10.1016/j.neuron.2010.11.002.

[35] Michael J. Berry, Iman H. Brivanlou, Thomas A. Jordan, and Markus Meister. Anticipation of moving stimuli by the retina. Nature, 398(6725):334–338, 1999. doi: 10.1038/18678.

[36] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2Act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision (ECCV), 2024.

[37] Raunaq Bhirangi, Tess Hellebrekers, Carmel Majidi, and Abhinav Gupta. ReSkin: Versatile, replaceable, lasting tactile skins. In Proceedings ofthe 5th Conference on Robot Learning (CoRL), 2021.

[38] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[39] James W. Bisley and Michael E. Goldberg. Attention, intention, and priority in the parietal lobe. Annual Review ofNeuroscience, 33:1–21, 2010. doi: 10.1146/annurev-neuro-060909-152823.

[40] Johan Bjorck et al. GR00T n1: An open foundation model for generalist humanoid robots. Technical report, NVIDIA, 2025.

[41] Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pretrained image-editing difusion models. arXiv preprint arXiv:2310.10639, 2023.

[42] Kevin Black, Noah Brown, Danny Driess, et al. �<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[43] Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[44] Sarah-Jayne Blakemore, Daniel M. Wolpert, and Chris D. Frith. Central cancellation of self-produced tickle sensation. Nature Neuroscience, 1(7):635–640, 1998. doi: 10.1038/2870.

[45] Jeannette Bohg, Karol Hausman, Bharath Sankaran, Oliver Brock, Danica Kragic, Stefan Schaal, and Gaurav S. Sukhatme. Interactive perception: Leveraging action in perception and perception in action. IEEE Transactions on Robotics, 33(6):1273–1291, 2017. doi: 10.1109/TRO.2017.2721939.

[46] Richard T. Born and David C. Bradley. Structure and function of visual area MT. Annual Review of Neuroscience, 28:157–189, 2005. doi: 10.1146/annurev.neuro.26.041002.131052.

[47] Andreea C. Bostan and Peter L. Strick. The basal ganglia and the cerebellum: Nodes in an integrated network. Nature Reviews Neuroscience, 19(6):338–350, 2018.

[48] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[49] Anthony Brohan, Noah Brown, Justice Carbajal, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[50] Antonia Bronars, Younghyo Park, and Pulkit Agrawal. Tune to learn: How controller gains shape robot policy learning. arXiv preprint arXiv:2604.02523, 2026.

[51] Spencer T. Brown, Mauricio Medina-Pizarro, Meghana Holla, Christopher E. Vaaga, and Indira M. Raman. Simple spike patterns and synaptic mechanisms encoding sensory and motor signals in Purkinje cells and the cerebellar nuclei. Neuron, 112(11):1848–1861, 2024. doi: 10.1016/j.neuron.2024.02.014.

[52] Jake Bruce et al. Genie: Generative interactive environments. Technical report, arXiv, 2024.

[53] Etienne Burdet, Rieko Osu, David W. Franklin, Theodore E. Milner, and Mitsuo Kawato. The central nervous system stabilizes unstable dynamics by learning optimal impedance. Nature, 414(6862):446–449, 2001. doi: 10.1038/35106566.

[54] Alejandra Harris Caceres, Deborah A. Barany, Neil M. Dundon, Jolinda Smith, and Michelle Marneweck. Neural encoding of direction and distance across reference frames in visually guided reaching. eNeuro, 11(12): ENEURO.0405–24.2024, 2024. doi: 10.1523/ENEURO.0405-24.2024.

[55] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9650–9660, 2021.

[56] Ilaria Carta, Christopher H. Chen, Amanda L. Schott, Schnaude Dorizan, and Kamran Khodakhah. Cerebellar modulation of the reward circuitry and social behavior. Science, 363(6424), January 2019. ISSN 1095-9203. doi: 10.1126/science.aav0581. URL http://dx.doi.org/10.1126/science.aav0581.

[57] N. Alex Cayco-Gajic and R. Angus Silver. Re-evaluating circuit mechanisms underlying pattern separation. Neuron, 101(4):584–602, 2019.

[58] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. WorldVLA: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.

[59] Nadia L. Cerminara, Richard Apps, and Dilwyn E. Marple-Horvat. An internal model of a moving visual target in the lateral cerebellum. The Journal of Physiology, 587(2):429–442, 2009. doi: 10.1113/jphysiol.2008.163337.

[60] Francois P. Chabrol, Antonin Blot, and Thomas D. Mrsic-Flogel. Cerebellar contribution to preparatory activity in motor neocortex. Neuron, 103(3):506–519, 2019. doi: 10.1016/j.neuron.2019.05.022.

[61] C. Elaine Chapman. Active versus passive touch: Factors influencing the transmission of somatosensory signals to primary somatosensory cortex. Canadian Journal ofPhysiology and Pharmacology, 72(5):558–570, 1994. doi: 10.1139/y94-080.

[62] Bingao Chen, Haoquan Fang, and C. Karen Liu. Astra on RoboMME: A three-tier planner, executor, and completion monitor. https://bingaochen.github.io/Astra-on-RoboMME/, 2026. Project page and code; accessed 3 October 2026.

[63] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. arXiv:2401.12168.

[64] Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, and Pheng-Ann Heng. Fast-in-slow: A dual-system VLA model unifying fast manipulation within slow reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[65] Jiaxin Chen, Hao Fang, Chen Wang, Shuo Wang, and Cewu Lu. History-aware visuomotor policy learning via point tracking. arXiv preprint arXiv:2509.17141, 2025.

[66] Kaiyuan Chen, Shuangyu Xie, Letian Fu, Justin Yu, William Pacini, Sandeep Bajamahal, Hudson Kim, Jaimyn Drake, Daehwa Kim, Haoru Xue, Jonathan Francis, Christian Juette, Peter Schaldenbrand, Muhammet Yunus Seker, Ruwan Wickramarachchi, Uksang Yoo, Guanzhi Wang, Adithyavairavan Murali, Balakumar Sundaralingam, S. Shankar Sastry, Spencer Huang, Yuke Zhu, Linxi Fan, and Ken Goldberg. GaP: A graph-as-policy multi-agent self-learning harness for variational automation tasks. arXiv preprint arXiv:2607.05369, 2026.

[67] Shuxiao Chen, Bike Zhang, Mark W. Mueller, Akshara Rai, and Koushil Sreenath. Learning torque control for quadrupedal locomotion. In IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2023.

[68] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. ConRFT: A reinforced fine-tuning method for VLA models via consistency policy. Robotics: Science and Systems XXI, 2025. doi: 10.15607/RSS.2025.XXI.019.

[69] Yuxuan Chen, Wanruo Zhang, and Xiao Li. Reflex: Enabling fast and predictive vision-language-action models for reaction-critical manipulation. arXiv preprint arXiv:2608.14379, 2026.

[70] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Difusion policy: Visuomotor policy learning via action difusion. In Robotics: Science and Systems, 2023.

[71] Ian Chuang, Andrew Lee, Dechen Gao, and Iman Soltani. Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation, 2024. arXiv:2409.17435.

[72] Mark M. Churchland and Krishna V. Shenoy. Preparatory activity and the expansive null-space. Nature Reviews Neuroscience, 25:213–236, 2024.

[73] Mark M. Churchland, John P. Cunningham, Matthew T. Kaufman, Justin D. Foster, Paul Nuyujukian, Stephen I. Ryu, and Krishna V. Shenoy. Neural population dynamics during reaching. Nature, 487(7405):51–56, 2012. doi: 10.1038/nature11129.

[74] Paul Cisek and John F. Kalaska. Neural correlates of reaching decisions in dorsal premotor cortex: Specification of multiple direction choices and final selection of action. Neuron, 45(5):801–814, 2005.

[75] Paul Cisek and John F. Kalaska. Neural mechanisms for interacting with a world full of action choices. Annual Review ofNeuroscience, 33:269–298, 2010.

[76] William T. Clower and Garrett E. Alexander. Movement sequence-related activity reflecting numerical order of components in supplementary and presupplementary motor areas. Journal ofNeurophysiology, 80(3):1562–1566, 1998.

[77] Carol L. Colby and Michael E. Goldberg. Space and attention in parietal cortex. Annual Review ofNeuroscience, 22:319–349, 1999.

[78] G. Corniani and H. P. Saal. Tactile innervation densities across the whole body. Journal of Neurophysiology, 124 (4):1229–1240, 2020. doi: 10.1152/jn.00313.2020.

[79] Thomas M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE Transactions on Electronic Computers, EC-14(3):326–334, 1965. doi: 10.1109/PGEC. 1965.264137.

[80] J. Douglas Crawford, Denise Y. P. Henriques, and W. Pieter Medendorp. Three-dimensional transformations for goal-directed action. Annual Review of Neuroscience, 34:309–331, 2011.

[81] Can Cui, Pengxiang Ding, Wenxuan Song, et al. OpenHelix: A short survey, empirical analysis, and open-source dual-system VLA model for robotic manipulation. arXiv preprint arXiv:2505.03912, 2025.

[82] Christine A. Curcio and Kimberly A. Allen. Topography of ganglion cells in human retina. Journal ofComparative Neurology, 300(1):5–25, 1990. doi: 10.1002/cne.903000103.

[83] Christine A. Curcio, Kenneth R. Sloan, Robert E. Kalina, and Anita E. Hendrickson. Human photoreceptor topography. Journal of Comparative Neurology, 292(4):497–523, 1990.

[84] Joshua Dacre, Matt Colligan, Thomas Clarke, Julian J. Ammer, Julia Schiemann, Victor Chamosa-Pino, Federico Claudi, J. Alex Harston, Constantinos Eleftheriou, Janelle M. P. Pakan, Cheng-Chiu Huang, Adam W. Hantman, Nathalie L. Rochefort, and Ian Duguid. A cerebellar-thalamocortical pathway drives behavioral context-dependent movement initiation. Neuron, 109(14):2326–2338, 2021. doi: 10.1016/j.neuron.2021.05.016.

[85] Ravinder S. Dahiya, Giorgio Metta, Maurizio Valle, and Giulio Sandini. Tactile sensing—from humans to humanoids. IEEE Transactions on Robotics, 26(1):1–20, 2010.

[86] Chris I. De Zeeuw. Bidirectional learning in upbound and downbound microzones of the cerebellum. Nature Reviews Neuroscience, 22(2):92–110, November 2021. ISSN 1471-0048. doi: 10.1038/s41583-020-00392-x. URL http://dx.doi.org/10.1038/s41583-020-00392-x

[87] Michel Desmurget and Scott T. Grafton. Forward modeling allows feedback control for fast reaching movements. Trends in Cognitive Sciences, 4(11):423–431, 2000. doi: 10.1016/S1364-6613(00)01537-0.

[88] Heiner Deubel and Werner X. Schneider. Saccade target selection and object recognition: Evidence for a common attentional mechanism. Vision Research, 36(12):1827–1837, 1996. doi: 10.1016/0042-6989(95)00294-4.

[89] Ashesh K. Dhawale, Maurice A. Smith, and Bence P. Ölveczky. The role of variability in motor learning. Annual Review ofNeuroscience, 40:479–498, 2017. doi: 10.1146/annurev-neuro-072116-031548.

[90] Ashesh K. Dhawale, Yohsuke R. Miyamoto, Maurice A. Smith, and Bence P. Ölveczky. Adaptive regulation of motor variability. Current Biology, 29(21):3551–3562, 2019.

[91] Ashesh K. Dhawale, Stefen B. E. Wolf, Raymond Ko, and Bence P. Ölveczky. The basal ganglia control the detailed kinematics of learned motor skills. Nature Neuroscience, 24(9):1256–1269, July 2021. ISSN 1546-1726. doi: 10.1038/s41593-021-00889-3. URL http://dx.doi.org/10.1038/s41593-021-00889-3.

[92] Vaggelis Dorovatas, Malte Schwerin, Andrew D. Bagdanov, Lucas Caccia, Antonio Carta, Laurent Charlin, Barbara Hammer, Tyler L. Hayes, Timm Hess, Christopher Kanan, Dhireesha Kudithipudi, Xialei Liu, Vincenzo Lomonaco, Jorge Mendez-Mendez, Darshan Patil, Ameya Prabhu, Elisa Ricci, Tinne Tuytelaars, Gido M. van de Ven, Liyuan Wang, Joost van de Weijer, Jonghyun Choi, Martin Mundt, and Rahaf Aljundi. Modular memory is the key to continual learning agents. arXiv preprint arXiv:2603.01761, 2026.

[93] Ria Doshi, Homer Walke, Oier Mees, Sudeep Dasari, and Sergey Levine. Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation. In Conference on Robot Learning (CoRL), 2024.

[94] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[95] Julien Doyon, Virginia Penhune, and Leslie G. Ungerleider. Distinct contribution of the cortico-striatal and cortico-cerebellar systems to motor skill learning. Neuropsychologia, 41(3):252–262, 2003. doi: 10.1016/ S0028-3932(02)00158-6.

[96] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, and Jonathan Tompson. Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[97] Charles J. Dufy and Robert H. Wurtz. Sensitivity of MST neurons to optic flow stimuli. I. a continuum of response selectivity to large-field stimuli. Journal of Neurophysiology, 65(6):1329–1345, 1991. doi: 10.1152/jn.1991.65.6.1329.

[98] Jean-René Duhamel, Carol L. Colby, and Michael E. Goldberg. The updating of the representation of visual space in parietal cortex by intended eye movements. Science, 255:90–92, 1992.

[99] Jean-René Duhamel, Frank Bremmer, Suliann Ben Hamed, and Werner Graf. Spatial invariance of visual receptive fields in parietal cortex neurons. Nature, 389:845–848, 1997. doi: 10.1038/39865.

[100] Richard P. Dum and Peter L. Strick. The origin of corticospinal projections from the premotor areas in the frontal lobe. Journal ofNeuroscience, 11(3):667–689, 1991.

[101] Dyna Robotics. Dyna-2: A 1-million-hour scaling law for world-action models. https://www.dyna.co/ dyna-2, August 2026. Technical blog post.

[102] ThankGod Egbe, Peng Wang, Zhihao Guo, Zidong Chen, et al. DINOv3-difusion policy: Self-supervised large visual model for visuomotor difusion policy learning. arXiv preprint arXiv:2509.17684, 2025.

[103] Roger M. Enoka. Neuromechanics ofHuman Movement. Human Kinetics, Champaign, IL, 5th edition, 2015.

[104] Marc O. Ernst and Martin S. Banks. Humans integrate visual and haptic information in a statistically optimal fashion. Nature, 415(6870):429–433, 2002. doi: 10.1038/415429a.

[105] Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, and Cewu Lu. AnyGrasp: Robust and eficient grasp perception in spatial and temporal domains. IEEE Transactions on Robotics, 39(5):3929–3945, 2023. doi: 10.1109/TRO.2023.3281153.

[106] Patrizia Fattori, Rossella Breveglieri, Nicoletta Marzocchi, Daniela Filippini, Annalisa Bosco, and Claudio Galletti. Hand orientation during reach-to-grasp movements modulates neuronal activity in the medial posterior parietal area V6A. The Journal ofNeuroscience, 29(6):1928–1936, 2009. doi: 10.1523/JNEUROSCI.4998-08.2009.

[107] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research, 23:1–39, 2022. URL https: //mlanthology.org/jmlr/2022/fedus2022jmlr-switch/.

[108] Ruoxuan Feng, Jiangyu Hu, Wenke Xia, Tianci Gao, Ao Shen, Yuhao Sun, Bin Fang, and Di Hu. AnyTouch: Learning unified static-dynamic representation across multiple visuo-tactile sensors. In International Conference on Learning Representations (ICLR), 2025. arXiv:2502.12191.

[109] Figure AI. Helix: A vision-language-action model for generalist humanoid control. https://www.figure.ai/ news/helix, 2025. Company technical report.

[110] Jeremy A. Fishel and Gerald E. Loeb. Bayesian exploration for intelligent identification of textures. Frontiers in Neurorobotics, 6:4, 2012. doi: 10.3389/fnbot.2012.00004.

[111] Jeremy A. Fishel and Gerald E. Loeb. Sensing tactile microvibrations with the BioTac — comparison with human sensitivity. In 2012 4th IEEE RAS & EMBS International Conference on Biomedical Robotics and Biomechatronics (BioRob), pages 1122–1127, Rome, Italy, 2012. doi: 10.1109/BioRob.2012.6290741.

[112] J. Randall Flanagan, Miles C. Bowman, and Roland S. Johansson. Control strategies in object manipulation tasks. Current Opinion in Neurobiology, 16(6):650–659, 2006. doi: 10.1016/j.conb.2006.10.005.

[113] Leonardo Fogassi, Vittorio Gallese, Luciano Fadiga, Giuseppe Luppino, Massimo Matelli, and Giacomo Rizzolatti. Coding of peripersonal space in inferior premotor cortex (area F4). Journal of Neurophysiology, 76 (1):141–157, 1996.

[114] Franka Robotics GmbH. Franka control interface documentation. https://frankaemika.github.io/docs/, 2023.

[115] David W. Franklin, Etienne Burdet, Keng Peng Tee, Rieko Osu, Chee-Meng Chew, Theodore E. Milner, and Mitsuo Kawato. CNS learns stable, accurate, and eficient movements using a simple algorithm. Journal of Neuroscience, 28(44):11165–11173, 2008. doi: 10.1523/JNEUROSCI.3099-08.2008.

[116] Nicolas Frémaux and Wulfram Gerstner. Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. Frontiers in Neural Circuits, 9:85, 2016. doi: 10.3389/fncir.2015.00085.

[117] Fu, Huang, Datta, Chen, Panitch, Liu, Li, and Goldberg. In-context imitation learning via next-token prediction, 2024. ICRT.

[118] Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Dantong Niu, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, and Linxi Fan. CaP-X: A framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435, 2026.

[119] Hirofumi Fujita, Takashi Kodama, and Sascha du Lac. Modular output circuits of the fastigial nucleus for diverse motor and nonmotor functions of the cerebellar vermis. eLife, 9:e58613, 2020. doi: 10.7554/eLife.58613.

[120] Galbot Team, Xuchuan Chen, Xiaoqian Cheng, Yu Deng, Lihe Ding, Shaocong Dong, Xiangjun Gao, Haozhe Jia, Zekai Li, Zhoujian Li, Yunrui Lian, Sikai Liang, Chenghuai Lin, Dairu Liu, Jiahang Liu, Qingtao Liu, Yuxuan Ma, Zekun Qi, Jiayi Su, He Wang, Ruochen Xu, Tianyu Xu, Xudong Xu, Zhe Xu, Mi Yan, Siming Yan, Li Yi, Ruixi Yu, Jinlu Zhang, Yintianrun Zhang, Zhikai Zhang, Zhizheng Zhang, Yixin Zheng, and Weiyi Zhu. Systematically exploring the capabilities of GPT-6 Astra as embodied policies. arXiv preprint arXiv:2609.38537, 2026.

[121] Claudio Galletti, Dieter F. Kutz, Michela Gamberini, Rossella Breveglieri, and Patrizia Fattori. Role of the medial parieto-occipital cortex in the control of reaching and grasping movements. Experimental Brain Research, 153(2):158–170, 2003. doi: 10.1007/s00221-003-1589-z.

[122] Zhenyu Gao, Courtney Davis, Alyse M. Thomas, Michael N. Economo, Amada M. Abrego, Karel Svoboda, Chris I. De Zeeuw, and Nuo Li. A cortico-cerebellar loop for motor planning. Nature, 563(7729):113–116, 2018. doi: 10.1038/s41586-018-0633-x.

[123] Gemini Robotics Team, Google DeepMind. Gemini robotics: Bringing AI into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[124] Mikhail Genkin, Krishna V. Shenoy, Chandramouli Chandrasekaran, and Tatiana A. Engel. The dynamics and geometry of choice in the premotor cortex. Nature, 645:168–176, 2025.

[125] Apostolos P. Georgopoulos, Andrew B. Schwartz, and Ronald E. Kettner. Neuronal population coding of movement direction. Science, 233(4771):1416–1419, 1986.

[126] Wulfram Gerstner, Marco Lehmann, Vasiliki Liakoni, Dane Corneil, and Johanni Brea. Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. Frontiers in Neural Circuits, 12:53, 2018. doi: 10.3389/fncir.2018.00053.

[127] James J. Gibson. Observations on active touch. Psychological Review, 69(6):477–491, 1962. doi: 10.1037/ h0046962.

[128] A. Gonshor and G. Melvill Jones. Extreme vestibulo-ocular adaptation induced by prolonged optical reversal of vision. The Journal ofPhysiology, 256(2):381–414, 1976.

[129] Melvyn A. Goodale and A. David Milner. Separate visual pathways for perception and action. Trends in Neurosciences, 15(1):20–25, 1992.

[130] Melvyn A. Goodale, Denis Pélisson, and Claude Prablanc. Large adjustments in visually guided reaching do not depend on vision of the hand or perception of target displacement. Nature, 320:748–750, 1986. doi: 10.1038/320748a0.

[131] Melvyn A. Goodale, A. David Milner, Lorna S. Jakobson, and David P. Carey. A neurological dissociation between perceiving objects and grasping them. Nature, 349:154–156, 1991.

[132] A. M. Gordon, A. F. Huxley, and F. J. Julian. The variation in isometric tension with sarcomere length in vertebrate muscle fibres. The Journal of Physiology, 184(1):170–192, 1966. doi: 10.1113/jphysiol.1966.sp007909.

[133] Ethan K. Gordon, Bruke Baraki, Hien Bui, and Michael Posa. Active tactile exploration for rigid body pose and shape estimation. arXiv preprint arXiv:2510.13595, 2025.

[134] Ann M. Graybiel. The basal ganglia and chunking of action repertoires. Neurobiology of Learning and Memory, 70(1–2):119–136, 1998.

[135] Michael S. A. Graziano. Ethological action maps: A paradigm shift for the motor cortex. Trends in Cognitive Sciences, 20(2):121–132, 2016.

[136] Michael S. A. Graziano, Gregory S. Yap, and Charles G. Gross. Coding of visual space by premotor neurons. Science, 266(5187):1054–1057, 1994.

[137] Michael S. A. Graziano, Xin Tian Hu, and Charles G. Gross. Visuospatial properties of ventral premotor cortex. Journal ofNeurophysiology, 77(5):2268–2292, 1997.

[138] Michael S. A. Graziano, Charlotte S. R. Taylor, and Tirin Moore. Complex movements evoked by microstimulation of precentral cortex. Neuron, 34(5):841–851, 2002.

[139] Sten Grillner. Control of locomotion in bipeds, tetrapods, and fish. In Handbook ofPhysiology, The Nervous System II, pages 1179–1236. American Physiological Society, 1981.

[140] Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, and Florian Shkurti. Safe: Multitask failure detection for vision-language-action models. In Advances in Neural Information Processing Systems 38, NeurIPS 2025, page 44801–44836. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025. doi: 10.52202/085713-1337. URL http://dx.doi.org/10.52202/085713-1337.

[141] Tanuj Gulati, Ling Guo, Dhakshin S. Ramanathan, Anitha Bodepudi, and Karunesh Ganguly. Neural reactivations during sleep determine network credit assignment. Nature Neuroscience, 20(9):1277–1284, 2017.

[142] Mohan W. Gupta and Timothy C. Rickard. Dissipation of reactive inhibition is suficient to explain post-rest improvements in motor sequence learning. npj Science ofLearning, 7(1), October 2022. ISSN 2056-7936. doi: 10.1038/s41539-022-00140-z. URL http://dx.doi.org/10.1038/s41539-022-00140-z.

[143] David Ha and Jürgen Schmidhuber. World models. In Advances in Neural Information Processing Systems, 2018.

[144] Huy Ha, Pete Florence, and Shuran Song. Scaling up and distilling down: Language-guided robot skill acquisition. arXiv preprint arXiv:2307.14535, 2023.

[145] Sami Haddadin, Alessandro De Luca, and Alin Albu-Schäfer. Robot collisions: A survey on detection, isolation, and identification. IEEE Transactions on Robotics, 33(6):1292–1312, 2017. doi: 10.1109/TRO.2017.2723903.

[146] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. Technical report, arXiv, 2023.

[147] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, April 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-08744-2. URL http://dx.doi.org/10.1038/s41586-025-08744-2.

[148] Nicklas Hansen, Xiaolong Wang, and Hao Su. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, 2024.

[149] Peng Hao, Chaofan Zhang, Dingzhe Li, Xiaoge Cao, Xiaoshuai Hao, Shaowei Cui, and Shuo Wang. TLA: Tactile-language-action model for contact-rich manipulation. arXiv preprint arXiv:2503.08548, 2025.

[150] Mary M. Hayhoe. Vision and action. Annual Review of Vision Science, 3:389–413, 2017. doi: 10.1146/ annurev-vision-102016-061437.

[151] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. arXiv:2111.06377.

[152] Sida He, Lingxi Xie, Yunning Cao, Pengfei Chen, Kaiwen Duan, Jiannan Ge, Xinyue Huo, Jiacheng Shao, and Qi Tian. Robot manipulation with GPT-6-Astra: Body knowledge, experience reuse, emergent skills, and Sim2Real transfer. arXiv preprint arXiv:2609.31770, 2026.

[153] Elwood Henneman, George Somjen, and David O. Carpenter. Functional significance of cell size in spinal motoneurons. Journal of Neurophysiology, 28(3):560–580, 1965. doi: 10.1152/jn.1965.28.3.560.

[154] Suzana Herculano-Houzel. The human brain in numbers: A linearly scaled-up primate brain. Frontiers in Human Neuroscience, 3:31, 2009.

[155] Carolina Higuera, Akash Sharma, Chaithanya Krishna Bodduluri, Taosha Fan, Patrick Lancaster, Mrinal Kalakrishnan, Michael Kaess, Byron Boots, Mike Lambeta, Tingfan Wu, and Mustafa Mukadam. Sparsh: Self-supervised touch representations for vision-based tactile sensing. In Conference on Robot Learning (CoRL), 2024. arXiv:2410.24090.

[156] Carolina Higuera, Sergio Arnaud, Byron Boots, Mustafa Mukadam, Francois Robert Hogan, and Franziska Meier. Visuo-tactile world models. arXiv preprint arXiv:2602.06001, 2026.

[157] Okihide Hikosaka, Hiroyuki Nakahara, Miya K. Rand, Katsuyuki Sakai, Xiaofeng Lu, Kae Nakamura, Shigehiro Miyachi, and Kenji Doya. Parallel neural networks for learning sequential procedures. Trends in Neurosciences, 22(10):464–471, 1999. doi: 10.1016/S0166-2236(99)01439-3.

[158] Archibald V. Hill. The heat of shortening and the dynamic constants of muscle. Proceedings ofthe Royal Society ofLondon. Series B, 126(843):136–195, 1938. doi: 10.1098/rspb.1938.0050.

[159] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 6840–6851, 2020.

[160] Neville Hogan. Adaptive control of mechanical impedance by coactivation of antagonist muscles. IEEE Transactions on Automatic Control, 29(8):681–690, 1984. doi: 10.1109/TAC.1984.1103644.

[161] Neville Hogan. Impedance control: An approach to manipulation: Part I—theory. Journal ofDynamic Systems, Measurement, and Control, 107(1):1–7, 1985. doi: 10.1115/1.3140702.

[162] Alex O. Holcombe. Attending to Moving Objects. Elements in Perception. Cambridge University Press, 2023. doi: 10.1017/9781009003414.

[163] Ryan Hoque, Peide Huang, David Milam Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

[164] Yifan Hou, Zeyi Liu, Cheng Chi, Eric Cousineau, Naveen Kuppuswamy, Siyuan Feng, Benjamin Burchfiel, and Shuran Song. Adaptive compliance policy: Learning approximate compliance for difusion guided control. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

[165] Robert D. Howe and Mark R. Cutkosky. Dynamic tactile sensing: Perception of fine surface features with stress rate sensing. IEEE Transactions on Robotics and Automation, 9(2):140–151, 1993. doi: 10.1109/70.238279.

[166] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. arXiv:2106.09685.

[167] Jiaheng Hu, Jay Shim, Chen Tang, et al. Simple recipe works: Vision-language-action models are natural continual learners with reinforcement learning. arXiv preprint arXiv:2603.11653, 2026.

[168] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

[169] Binghao Huang, Yixuan Wang, Xinyi Yang, Yiyue Luo, and Yunzhu Li. 3D-ViTac: Learning fine-grained manipulation with visuo-tactile sensing. In Conference on Robot Learning (CoRL), 2024.

[170] Jialei Huang, Shuo Wang, Fanqi Lin, Yihang Hu, Chuan Wen, and Yang Gao. Tactile-VLA: Unlocking vision-language-action model’s physical knowledge for tactile generalization. arXiv preprint arXiv:2507.09160, 2025.

[171] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. VoxPoser: Composable 3D value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.

[172] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. ReKep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024.

[173] Marco Hutter, Christian Gehring, Dominic Jud, Andreas Lauber, C. Dario Bellicoso, Vassilios Tsounis, Jemin Hwangbo, Karen Bodie, Peter Fankhauser, Michael Bloesch, Remo Diethelm, Samuel Bachmann, Amir Melzer, and Mark Hoepflinger. ANYmal — a highly mobile and dynamic quadrupedal robot. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 38–44, 2016.

[174] Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics, 4(26):eaau5872, 2019.

[175] Hidehiko K. Inagaki, Susu Chen, Margreet C. Ridder, Pankaj Sah, Nuo Li, Zidan Yang, Hana Hasanbegovic, Zhenyu Gao, Chris I. De Zeeuw, and Karel Svoboda. A midbrain-thalamus-cortex circuit reorganizes cortical dynamics to initiate movement. Cell, 185(6):1065–1081, 2022.

[176] Iole Indovina, Vincenzo Mafei, Gianfranco Bosco, Myrka Zago, Emiliano Macaluso, and Francesco Lacquaniti. Representation of visual gravitational motion in the human vestibular cortex. Science, 308(5720):416–419, 2005. doi: 10.1126/science.1107961.

[177] Atsushi Iriki, Michio Tanaka, and Yoshiaki Iwamura. Coding of modified body schema during tool use by macaque postcentral neurones. NeuroReport, 7(14):2325–2330, 1996.

[178] Masaki Isoda and Okihide Hikosaka. Switching from automatic to controlled action by monkey medial frontal cortex. Nature Neuroscience, 10(2):240–248, 2007.

[179] Masao Ito. Cerebellar control of the vestibulo-ocular reflex—around the flocculus hypothesis. Annual Review of Neuroscience, 5:275–296, 1982.

[180] Masao Ito. Cerebellar long-term depression: Characterization, signal transduction, and functional roles. Physiological Reviews, 81(3):1143–1195, 2001.

[181] Masao Ito. Control of mental activities by internal models in the cerebellum. Nature Reviews Neuroscience, 9(4): 304–313, 2008.

[182] Hongjin Ji, Guoyang Xia, Luoyang Sun, Fangxiang Feng, and Lei Ren. Vane: Reliable test-time training for vision-language-action models via future visual representation prediction. arXiv preprint arXiv:2608.09448, 2026.

[183] Tianchong Jiang, Jingtian Ji, Xiangshan Tan, Jiading Fang, Anand Bhattad, Vitor Guizilini, and Matthew R. Walter. Do you know where your camera is? view-invariant policy learning with camera conditioning. In IEEE International Conference on Robotics and Automation (ICRA), 2026.

[184] Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, and Linxi Fan. RoboTTT: Context scaling for robot policies. arXiv preprint arXiv:2607.15275, 2026.

[185] M. S. Jog, Y. Kubota, C. I. Connolly, V. Hillegaart, and A. M. Graybiel. Building neural representations of habits. Science, 286(5445):1745–1749, 1999.

[186] R. S. Johansson and Å. B. Vallbo. Tactile sensibility in the human hand: relative and absolute densities of four types of mechanoreceptive units in glabrous skin. The Journal of Physiology, 286:283–300, 1979. doi: 10.1113/jphysiol.1979.sp012619.

[187] R. S. Johansson and G. Westling. Roles of glabrous skin receptors and sensorimotor memory in automatic control of precision grip when lifting rougher or more slippery objects. Experimental Brain Research, 56(3):550–564, 1984. doi: 10.1007/BF00237997.

[188] Roland S. Johansson and J. Randall Flanagan. Coding and use of tactile signals from the fingertips in object manipulation tasks. Nature Reviews Neuroscience, 10(5):345–359, 2009.

[189] Roland S. Johansson and Göran Westling. Signals in tactile aferents from the fingers eliciting adaptive motor responses during precision grip. Experimental Brain Research, 66(1):141–154, 1987. doi: 10.1007/BF00236210.

[190] Roland S. Johansson and Göran Westling. Programmed and triggered actions to rapid load changes during precision grip. Experimental Brain Research, 71(1):72–86, 1988. doi: 10.1007/BF00247523.

[191] Roland S. Johansson, Christer Häger, and Lennart Bäckström. Somatosensory control of precision grip during unpredictable pulling loads. III. impairments during digital anesthesia. Experimental Brain Research, 89(1): 204–213, 1992. doi: 10.1007/BF00229017.

[192] Roland S. Johansson, Ronald Riso, Charlotte Häger, and Lars Bäckström. Somatosensory control of precision grip during unpredictable pulling loads. I. changes in load force amplitude. Experimental Brain Research, 89(1): 181–191, 1992. doi: 10.1007/BF00229015.

[193] Roland S. Johansson, Göran Westling, Anders Bäckström, and J. Randall Flanagan. Eye–hand coordination in object manipulation. The Journal ofNeuroscience, 21(17):6917–6932, 2001. doi: 10.1523/JNEUROSCI. 21-17-06917.2001.

[194] K. O. Johnson. The roles and functions of cutaneous mechanoreceptors. Current Opinion in Neurobiology, 11(4): 455–461, 2001. doi: 10.1016/S0959-4388(00)00234-8.

[195] Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, New York, 2011.

[196] Rudolph E. Kalman. A new approach to linear filtering and prediction problems. Journal ofBasic Engineering, 82(1):35–45, 1960. doi: 10.1115/1.3662552.

[197] Eric R. Kandel, James H. Schwartz, Thomas M. Jessell, Steven A. Siegelbaum, and A. J. Hudspeth. Principles of Neural Science. McGraw-Hill, New York, 5th edition, 2013.

[198] S. Kastner, Q. Chen, S. K. Jeong, and R. E. B. Mruczek. A brief comparative review of primate posterior parietal cortex: A novel hypothesis on the human toolmaker. Neuropsychologia, 105:123–134, 2017. doi: 10.1016/j.neuropsychologia.2017.01.034.

[199] Matthew T. Kaufman, Mark M. Churchland, Stephen I. Ryu, and Krishna V. Shenoy. Cortical activity in the null space: Permitting preparation without movement. Nature Neuroscience, 17(3):440–448, 2014.

[200] Risa Kawai, Timothy Markman, Rajesh Poddar, Raymond Ko, Antoniu L. Fantana, Ashesh K. Dhawale, Adam R. Kampf, and Bence P. Ölveczky. Motor cortex is required for learning but not for executing a motor skill. Neuron, 86(3):800–812, 2015.

[201] Mitsuo Kawato. Internal models for motor control and trajectory planning. Current Opinion in Neurobiology, 9 (6):718–727, 1999.

[202] Mitsuo Kawato, Shogo Ohmae, Huu Hoang, and Terry Sanger. 50 years since the marr, ito, and albus models of the cerebellum. Neuroscience, 462:151–174, May 2021. ISSN 0306-4522. doi: 10.1016/j.neuroscience.2020.06.019. URL http://dx.doi.org/10.1016/j.neuroscience.2020.06.019.

[203] Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. HG-DAgger: Interactive imitation learning with human experts. In IEEE International Conference on Robotics and Automation (ICRA), pages 8077–8083, 2019.

[204] Roberta M. Kelly and Peter L. Strick. Cerebellar loops with motor cortex and prefrontal cortex of a nonhuman primate. The Journal ofNeuroscience, 23(23):8432–8444, 2003. doi: 10.1523/JNEUROSCI.23-23-08432.2003.

[205] Steve W. Kennerley, K. Sakai, and M. F. S. Rushworth. Organization of action sequences and the role of the pre-SMA. Journal ofNeurophysiology, 91(2):978–993, 2004. doi: 10.1152/jn.00651.2003.

[206] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023. arXiv:2308.04079.

[207] Konstantina Kilteni and H. Henrik Ehrsson. Functional connectivity between the cerebellum and somatosensory areas implements the attenuation of self-generated touch. The Journal ofNeuroscience, 40(4):894–906, 2020. doi: 10.1523/JNEUROSCI.1732-19.2019.

[208] Hyosub E. Kim, J. Ryan Morehead, Darius E. Parvin, Reza Moazzezi, and Richard B. Ivry. Invariant errors reveal limitations in motor correction rather than constraints on error sensitivity. Communications Biology, 1:19, 2018. doi: 10.1038/s42003-018-0021-y.

[209] Moo Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In Robotics: Science and Systems XXI, RSS2025. Robotics: Science and Systems Foundation, June 2025. doi: 10.15607/rss.2025.xxi.017. URL http://dx.doi.org/10.15607/rss.2025.xxi.017.

[210] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[211] Seungyeon Kim, Junhoo Lee, Minkyu Kim, Baekseung Kim, and Nojun Kwak. Recursive harness distillation across agents for robot manipulation. arXiv preprint arXiv:2609.33378, 2026.

[212] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy ofSciences, 114(13):3521–3526, 2017. doi: 10.1073/pnas.1611835114.

[213] Etienne Koechlin, Chrystèle Ody, and Frédérique Kouneiher. The architecture of cognitive control in the human prefrontal cortex. Science, 302(5648):1181–1185, 2003. doi: 10.1126/science.1088545.

[214] Dimitar Kostadinov, Maxime Beau, Marta Blanco-Pozo, and Michael Häusser. Predictive and reactive reward signals conveyed by climbing fiber inputs to cerebellar Purkinje cells. Nature Neuroscience, 22(6):950–962, 2019.

[215] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Ofline reinforcement learning with implicit Q-learning. In International Conference on Learning Representations (ICLR), 2022.

[216] Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. RMA: Rapid motor adaptation for legged robots. In Robotics: Science and Systems (RSS), 2021. arXiv:2107.04034

[217] Vaishak Kumar. Act-observe-rewrite: Multimodal coding agents as in-context policy learners for robot manipulation. arXiv preprint arXiv:2603.04466, 2026.

[218] Jun Kunimatsu, Tomoki W. Suzuki, Shogo Ohmae, and Masaki Tanaka. Diferent contributions of preparatory activity in the basal ganglia and cerebellum for self-timing. eLife, 7:e35676, 2018. doi: 10.7554/eLife.35676.

[219] Kenichi Kuriyama, Robert Stickgold, and Matthew P. Walker. Sleep-dependent learning and motor-skill complexity. Learning & Memory, 11(6):705–713, 2004. doi: 10.1101/lm.76304.

[220] Mike Lambeta, Po-Wei Chou, Stephen Tian, Brian Yang, Benjamin Maloon, Victoria Rose Most, Dave Stroud, Raymond Santos, Ahmad Byagowi, Gregg Kammerer, Dinesh Jayaraman, and Roberto Calandra. DIGIT: A novel design for a low-cost compact high-resolution tactile sensor with application to in-hand manipulation. IEEE Robotics and Automation Letters, 5(3):3838–3845, 2020. doi: 10.1109/LRA.2020.2977257.

[221] Michael F. Land and Mary Hayhoe. In what ways do eye movements contribute to everyday activities? Vision Research, 41(25–26):3559–3565, 2001. doi: 10.1016/S0042-6989(01)00102-X.

[222] Michael F. Land and Peter McLeod. From eye movements to actions: How batsmen hit the ball. Nature Neuroscience, 3(12):1340–1345, 2000. doi: 10.1038/81887.

[223] Michael F. Land and Benjamin W. Tatler. Looking and Acting: Vision and Eye Movements in Natural Behaviour. Oxford University Press, Oxford, 2009.

[224] Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, DJ Strouse, Steven Hansen, Angelos Filos, Ethan Brooks, Maxime Gazeau, Himanshu Sahni, Satinder Singh, and Volodymyr Mnih. In-context reinforcement learning with algorithm distillation. arXiv preprint arXiv:2210.14215, 2022. ICLR 2023.

[225] A. M. Lassek. The human pyramidal tract. IV. a study of the mature, myelinated fibers of the pyramid. Journal ofComparative Neurology, 76(2):217–225, 1942.

[226] A. M. Lassek and G. L. Rasmussen. The human pyramidal tract: A fiber and numerical analysis. Archives of Neurology & Psychiatry, 42(5):872–876, 1939.

[227] Susan J. Lederman and Roberta L. Klatzky. Hand movements: A window into haptic object recognition. Cognitive Psychology, 19(3):342–368, 1987. doi: 10.1016/0010-0285(87)90008-9.

[228] David N. Lee. A theory of visual control of braking based on information about time-to-collision. Perception, 5 (4):437–459, 1976. doi: 10.1068/p050437.

[229] Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, and Huazhe Xu. Performant robotic manipulation with real-world reinforcement learning. Science Robotics, 11 (116), July 2026. ISSN 2470-9476. doi: 10.1126/scirobotics.aed6267. URL http://dx.doi.org/10.1126/ scirobotics.aed6267.

[230] Roger N. Lemon. Descending pathways in motor control. Annual Review of Neuroscience, 31:195–218, 2008. doi: 10.1146/annurev.neuro.31.060407.125547.

[231] Peter Lennie. The cost of cortical computation. Current Biology, 13(6):493–497, 2003. doi: 10.1016/ S0960-9822(03)00135-0.

[232] Frank T. M. Leoné, Simona Monaco, Denise Y. P. Henriques, Ivan Toni, and W. Pieter Medendorp. Flexible reference frames for grasp planning in human parietofrontal cortex. eNeuro, 2(3):ENEURO.0008–15.2015, 2015. doi: 10.1523/ENEURO.0008-15.2015.

[233] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Ofline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

[234] Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, and Weihua Su. VLA-RFT: Vision-language-action reinforcement fine-tuning with verified rewards in world simulators. arXiv preprint arXiv:2510.00406, 2025.

[235] Qixiu Li, Yaobo Liang, Zeyu Wang, et al. CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

[236] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. In Robotics: Science and Systems XXI, RSS2025. Robotics: Science and Systems Foundation, June 2025. doi: 10.15607/rss.2025.xxi.074. URL http://dx.doi.org/10.15607/RSS.2025.XXI.074.

[237] Yang Li, Zhaxizhuoma, Hongru Jiang, Junjie Xia, Hongquan Zhang, Jinda Du, Yunsong Zhou, Jia Zeng, Ce Hao, Jieji Ren, Qiaojun Yu, Cewu Lu, Yu Qiao, and Jiangmiao Pang. ForceVLA2: Unleashing hybrid force-position control with force awareness for contact-rich manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[238] Ziang Li, Dongzhou Cheng, Yibin Wang, Shiyue Wang, Xiaoyang Xu, Lingxuan Weng, Juan Wang, and Jiaqi Wang. Light-WAM: Eficient world action models with state-fusion action decoding, 2026. URL https://arxiv.org/abs/2606.08242.

[239] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500, 2023. doi: 10.1109/ICRA48891.2023.10160591.

[240] Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao. Data scaling laws in imitation learning for robotic manipulation. arXiv preprint arXiv:2410.18647, 2024. ICLR 2025.

[241] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[242] Stephen G. Lisberger. Visual guidance of smooth-pursuit eye movements: Sensation, action, and what happens in between. Neuron, 66(4):477–491, 2010. doi: 10.1016/j.neuron.2010.03.027.

[243] Changyu Liu, Yiyang Liu, Taowen Wang, Qiao Zhuang, James Chenhao Liang, Wenhao Yang, Renjing Xu, Qifan Wang, Dongfang Liu, and Cheng Han. On-the-fly VLA adaptation via test-time reinforcement learning. arXiv preprint arXiv:2601.06748, 2026.

[244] Fangcheng Liu, Yeqing Shen, Anda Cheng, Weishi Mi, Chao Tang, Chenyuan Liu, Yushun Xiang, Tingguang Li, Yong-Lu Li, and Yehui Tang. RoboICL: Embodied in-context learning with GPT-6 Astra. arXiv preprint arXiv:2609.34261, 2026.

[245] Haowen Liu, Xirui Li, Shaoxiong Yao, Peng Shi, Tianyi Zhou, Jia-Bin Huang, Furong Huang, and Jiayuan Mao. Guava: Distilling frontier VLMs into a compact agent through a robotic manipulation harness. arXiv preprint arXiv:2606.18363, 2026.

[246] Huihan Liu, Changyeon Kim, Bo Liu, Minghuan Liu, and Yuke Zhu. Pretrained vision-language-action models are surprisingly resistant to forgetting in continual learning. arXiv preprint arXiv:2603.03818, 2026.

[247] Jason Jingzhou Liu, Yulong Li, Kenneth Shaw, Tony Tao, Ruslan Salakhutdinov, and Deepak Pathak. FACTR: Force-attending curriculum training for contact-rich policy learning. In Robotics: Science and Systems (RSS), 2025.

[248] Min Liu, Deepak Pathak, and Ananye Agarwal. LocoFormer: Generalist locomotion via long-context adaptation. In Conference on Robot Learning (CoRL), 2025. arXiv:2509.23745.

[249] Shengbang Liu, Yueru Jia, Yuyang Yan, Jiaming Liu, Xinran Zhang, Qiuxuan Feng, Yandong Guo, Shiji Zhou, Boxin Shi, and Shanghang Zhang. TACO: Tactile world model as a self-corrector for scalable VLA post-training. arXiv preprint arXiv:2607.02840, 2026.

[250] Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, and Chelsea Finn. Bidirectional decoding: Improving action chunking via guided test-time sampling. arXiv preprint arXiv:2408.17355, 2024.

[251] Zhe Liu, Jinghua Hou, Yuxiang Lu, Zhenya Yang, Xianzhe Fan, Junwei Luo, Junyi Li, Ruihua Han, Zhi Hou, and Hengshuang Zhao. StreamPI: Streaming multimodal temporal modeling for vision-language-action models. arXiv preprint arXiv:2608.26067, 2026.

[252] Aloysius Y. T. Low, Ayesha R. Thanawalla, Alaric K. K. Yip, Jinsook Kim, Kelly L. L. Wong, Martesa Tantra, George J. Augustine, and Albert I. Chen. Precision of discrete and rhythmic forelimb movements requires a distinct neuronal subpopulation in the interposed anterior nucleus. Cell Reports, 22(9):2322–2333, 2018. doi: 10.1016/j.celrep.2018.02.017.

[253] Sebastián Loyola, Tycho M Hoogland, Hugo Hoedemaker, Vincenzo Romano, Mario Negrello, and Chris I De Zeeuw. How inhibitory and excitatory inputs gate output of the inferior olive. eLife, 12, August 2023. ISSN 2050-084X. doi: 10.7554/elife.83239. URL http://dx.doi.org/10.7554/eLife.83239.

[254] Jianlan Luo, Charles Xu, Jefrey Wu, and Sergey Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. arXiv preprint arXiv:2410.21845, 2024.

[255] Shan Luo, Nathan F. Lepora, Wenzhen Yuan, Kaspar Althoefer, Gordon Cheng, and Ravinder Dahiya. Tactile robotics: An outlook. IEEE Transactions on Robotics, 2025.

[256] Chong Ma, Taiyi Su, Jian Zhu, Jianjun Zhang, Zitai Huang, Yi Xu, and Hanli Wang. PiL-World: A chunk-wise world model for VLA policy-in-the-loop evaluation. arXiv preprint arXiv:2606.05773, 2026.

[257] Yunchao Ma, Yizhuang Zhou, Yunhuan Yang, Tiancai Wang, and Haoqiang Fan. Running VLAs at real-time speed. arXiv preprint arXiv:2510.26742, 2025.

[258] David J. C. MacKay. Information-based objective functions for active data selection. Neural Computation, 4(4): 590–604, 1992. doi: 10.1162/neco.1992.4.4.590.

[259] Hung Phan Quoc Mai and Naeem Ul Islam. Mixture-of-experts action chunking transformers for high-precision robot imitation learning. Robotics andAutonomous Systems, 205:105616, 2026. doi: 10.1016/j.robot.2026.105616.

[260] David Marr. A theory of cerebellar cortex. The Journal of Physiology, 202(2):437–470, 1969.

[261] Peter B. C. Matthews. The human stretch reflex and the motor cortex. Trends in Neurosciences, 14(3):87–91, 1991. doi: 10.1016/0166-2236(91)90064-2.

[262] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology ofLearning and Motivation, volume 24, pages 109–165. Academic Press, 1989. doi: 10.1016/S0079-7421(08)60536-8.

[263] Samuel D. McDougle, Krista M. Bond, and Jordan A. Taylor. Explicit and implicit processes constitute the fast and slow processes of sensorimotor learning. Journal ofNeuroscience, 35(26):9568–9579, 2015.

[264] Joseph McIntyre, Myrka Zago, Alain Berthoz, and Francesco Lacquaniti. Does the brain model Newton’s laws? Nature Neuroscience, 4(7):693–694, 2001. doi: 10.1038/89477.

[265] Javier F. Medina, William L. Nores, and Michael D. Mauk. Inhibition of climbing fibres is a signal for the extinction of conditioned eyelid responses. Nature, 416(6878):330–333, 2002.

[266] Mark Van der Merwe and Devesh Jha. In-context iterative policy improvement for dynamic manipulation. arXiv preprint arXiv:2508.15021, 2025. CoRL 2025.

[267] Emily E. Meyer, Marcelina Martynek, Sabine Kastner, Margaret S. Livingstone, and Michael J. Arcaro. Expansion of a conserved architecture drives the evolution of the primate visual cortex. Proceedings ofthe National Academy ofSciences, 122(3):e2421585122, 2025. doi: 10.1073/pnas.2421585122.

[268] Hauke S. Meyerhof, Frank Papenmeier, and Markus Huf. Studying visual attention using the multiple object tracking paradigm: A tutorial review. Attention, Perception, & Psychophysics, 79(5):1255–1274, 2017. doi: 10.3758/s13414-017-1338-1.

[269] R. C. Miall, D. J. Weir, D. M. Wolpert, and J. F. Stein. Is the cerebellum a Smith predictor? Journal ofMotor Behavior, 25(3):203–216, 1993. doi: 10.1080/00222895.1993.9942050.

[270] Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, and Mingsheng Long. JEPA-VLA: Video predictive embedding is needed for VLA models. arXiv preprint arXiv:2602.11832, 2026.

[271] Earl K. Miller and Jonathan D. Cohen. An integrative theory of prefrontal cortex function. Annual Review of Neuroscience, 24:167–202, 2001.

[272] A. David Milner and Melvyn A. Goodale. Two visual systems re-viewed. Neuropsychologia, 46(3):774–785, 2008.

[273] H. S. Milner-Brown, R. B. Stein, and R. Yemm. The orderly recruitment of human motor units during voluntary isometric contractions. The Journal of Physiology, 230(2):359–370, 1973. doi: 10.1113/jphysiol.1973.sp010192.

[274] Jonathan W. Mink. The basal ganglia: focused selection and inhibition of competing motor programs. Progress in Neurobiology, 50(4):381–425, 1996

[275] Kevin G. C. Mizes, Jack Lindsey, G. Sean Escola, and Bence P. Ölveczky. The role of motor cortex in motor sequence execution depends on demands for flexibility. Nature Neuroscience, 27(12):2466–2475, November 2024. ISSN 1546-1726. doi: 10.1038/s41593-024-01792-3. URL http://dx.doi.org/10.1038/ s41593-024-01792-3.

[276] Joël Monzée, Yves Lamarre, and Allan M. Smith. The efects of digital anesthesia on force control using a precision grip. Journal ofNeurophysiology, 89(2):672–683, 2003. doi: 10.1152/jn.00434.2001.

[277] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), page 948–958. IEEE, March 2026. doi: 10.1109/wacv61042.2026.00099. URL http: //dx.doi.org/10.1109/WACV61042.2026.00099.

[278] Yao Mu, Junting Chen, Qinglong Zhang, Shoufa Chen, Qiaojun Yu, Chongjian Ge, Runjian Chen, Zhixuan Liang, Mengkang Hu, Chaofan Tao, Peize Sun, Haibao Yu, Chao Yang, Wenqi Shao, Wenhai Wang, Jifeng Dai, Yu Qiao, Mingyu Ding, and Ping Luo. RoboCodeX: Multimodal code generation for robotic behavior synthesis. arXiv preprint arXiv:2402.16117, 2024.

[279] Akira Murata, Luciano Fadiga, Leonardo Fogassi, Vittorio Gallese, Vassilis Raos, and Giacomo Rizzolatti. Object representation in the ventral premotor cortex (area F5) of the monkey. Journal ofNeurophysiology, 78(4): 2226–2230, 1997.

[280] Akira Murata, Vittorio Gallese, Giuseppe Luppino, Masakazu Kaseda, and Hideo Sakata. Selectivity for the shape, size, and orientation of objects for grasping in neurons of monkey parietal area AIP. Journal of Neurophysiology, 83(5):2580–2601, 2000.

[281] Wendy M. Murray, Scott L. Delp, and Thomas S. Buchanan. Variation of muscle moment arms with elbow and forearm position. Journal of Biomechanics, 28(5):513–525, 1995. doi: 10.1016/0021-9290(94)00114-J.

[282] Hajime Mushiake, Masahiko Inase, and Jun Tanji. Neuronal activity in the primate premotor, supplementary, and precentral motor cortex during visually guided and internally determined sequential movements. Journal of Neurophysiology, 66(3):705–718, 1991.

[283] Parashkev Nachev, Christopher Kennard, and Masud Husain. Functional role of the supplementary and pre-supplementary motor areas. Nature Reviews Neuroscience, 9:856–869, 2008.

[284] J. G. Nairn, K. S. Bedi, T. M. Mayhew, and L. F. Campbell. On the number of Purkinje cells in the human cerebellum: Unbiased estimates obtained by using the “fractionator”. Journal ofComparative Neurology, 290(4): 527–532, 1989. doi: 10.1002/cne.902900407.

[285] R. M. A. Napper and R. J. Harvey. Number of parallel fiber synapses on an individual Purkinje cell in the cerebellum of the rat. Journal ofComparative Neurology, 274(2):168–177, 1988. doi: 10.1002/cne.902740204.

[286] Soroush Nasiriany, Fei Xia, Wenhao Yu, Ted Xiao, Jacky Liang, Ishita Dasgupta, Annie Xie, Danny Driess, Ayzaan Wahid, Zhuo Xu, Quan Vuong, Tingnan Zhang, Tsang-Wei Edward Lee, Kuang-Huei Lee, Peng Xu, Sean Kirmani, Yuke Zhu, Andy Zeng, Karol Hausman, Nicolas Heess, Chelsea Finn, Sergey Levine, and Brian Ichter. PIVOT: Iterative visual prompting elicits actionable knowledge for vision-language models. In International Conference on Machine Learning, 2024.

[287] Zaruhi Navasardyan, Tatul Danielyan, and Hrant Davtyan. FailBench: How reliable are VLMs at judging robot task success? arXiv preprint arXiv:2609.03611, 2026.

[288] NeoteAI Team and Fudan TEAI Team. �<sub>0</sub>-VTLA: Scaling vision-tactile-language-action model with latent tactile tokens. arXiv preprint arXiv:2607.23782, 2026.

[289] Loc X. Nguyen, Avi Deb Raha, Huy Q. Le, Eui-Nam Huh, Dusit Niyato, and Choong Seon Hong. Self-evolving AI for humanoids: Mechanisms, safety, and evaluation of post-deployment self-improvement. arXiv preprint arXiv:2609.13236, 2026.

[290] Romi Nijhawan. Motion extrapolation in catching. Nature, 370(6487):256–257, 1994. doi: 10.1038/370256b0.

[291] Dennis A. Nowak, Joachim Hermsdörfer, Christian Marquardt, and Hans-Hermann Fuchs. Grip and load force coupling during discrete vertical arm movements with a grasped object in cerebellar atrophy. Experimental Brain Research, 145(1):28–39, 2002. doi: 10.1007/s00221-002-1079-8.

[292] NVIDIA. Cosmos world foundation models. Technical report, NVIDIA, 2025.

[293] Emily R. Oby, Matthew D. Golub, Jay A. Hennig, Alan D. Degenhart, Elizabeth C. Tyler-Kabara, Byron M. Yu, Steven M. Chase, and Aaron P. Batista. New neural activity patterns emerge with long-term learning. Proceedings ofthe National Academy ofSciences, 116(30):15210–15215, 2019. doi: 10.1073/pnas.1820296116.

[294] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In Robotics: Science and Systems (RSS), 2024.

[295] Shogo Ohmae and Javier F. Medina. Climbing fibers encode a temporal-diference prediction error during cerebellar learning in mice. Nature Neuroscience, 18(12):1798–1803, 2015.

[296] Abby O’Neill, Open X-Embodiment Collaboration, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

[297] OpenAI. Introducing GPT-6 Astra. https://openai.com/index/gpt-6-astra/, 2026. Company announcement, 3 September 2026; accessed 3 October 2026.

[298] OpenAI. Decisions. https://developers.openai.com/api/docs/guides/decisions, 2026. OpenAI API guide; accessed 25 October 2026.

[299] Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. arXiv:2304.07193.

[300] Jill X. O’Reilly, M. Marsel Mesulam, and Anna Christina Nobre. The cerebellum predicts the timing of perceptual events. The Journal ofNeuroscience, 28(9):2252–2260, 2008. doi: 10.1523/JNEUROSCI.2742-07.2008.

[301] Takayuki Osa, Joni Pajarinen, Gerhard Neumann, J. Andrew Bagnell, Pieter Abbeel, and Jan Peters. An algorithmic perspective on imitation learning. Foundations and Trends in Robotics, 7(1–2):1–179, 2018.

[302] Steven C. Pan and Timothy C. Rickard. Sleep and motor learning: Is there room for consolidation? Psychological Bulletin, 141(4):812–834, 2015. doi: 10.1037/bul0000009.

[303] Michel Paré, Catherine Behets, and Olivier Cornu. Paucity of presumptive Rufini corpuscles in the index finger pad of humans. Journal ofComparative Neurology, 456(3):260–266, 2003. doi: 10.1002/cne.10519.

[304] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. DeepSDF: Learning continuous signed distance functions for shape representation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 165–174, 2019.

[305] Junsung Park, Sachin Bhadang, Carmelo Sferrazza, Sha Yi, and Xiaolong Wang. TactX: Learning shared tactile representations across diverse sensors. arXiv preprint arXiv:2606.31236, 2026.

[306] Lloyd D. Partridge. Modifications of neural output signals by muscles: A frequency response study. Journal of Applied Physiology, 20(1):150–156, 1965. doi: 10.1152/jappl.1965.20.1.150.

[307] R. Peeters, L. Simone, K. Nelissen, M. Fabbri-Destro, W. Vandufel, G. Rizzolatti, and G. A. Orban. The representation of tool use in humans and monkeys: Common and uniquely human features. The Journal of Neuroscience, 29(37):11523–11539, 2009. doi: 10.1523/JNEUROSCI.2040-09.2009.

[308] Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In IEEE International Conference on Robotics and Automation (ICRA), pages 3803–3810, 2018.

[309] Abigail L. Person and Indira M. Raman. Purkinje neuron synchrony elicits time-locked spiking in the cerebellar nuclei. Nature, 481(7382):502–505, 2012.

[310] Karl Pertsch, Kyle Stachowicz, Brian Ichter, et al. FAST: Eficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[311] Chris D. Phillips, Alexander T. Hodge, Courtney C. Myers, Daniel K. Leventhal, and Christian R. Burgess. Striatal dopamine contributions to skilled motor learning. The Journal ofNeuroscience, 44(26):e0240242024, May 2024. ISSN 1529-2401. doi: 10.1523/jneurosci.0240-24.2024. URL http://dx.doi.org/10.1523/ JNEUROSCI.0240-24.2024.

[312] Physical Intelligence. $\pi _ { 0 . 7 } \colon$ A steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

[313] Physical Intelligence, Ali Amin, Raichelle Aniceto, et al. $\pi _ { 0 . 6 } ^ { * } { : }$ A VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025. RSS 2026.

[314] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Dibya Ghosh, Karol Hausman, Brian Ichter, Sergey Levine, Suraj Nair, Karl Pertsch, Allen Z. Ren, et al. �<sub>0.5</sub>: A vision-language-action model with open-world generalization. In Conference on Robot Learning (CoRL), 2025.

[315] C. Pichitpornchai, J. A. Rawson, and S. Rees. Morphology of parallel fibres in the cerebellar cortex of the rat: An experimental light and electron microscopic study with biocytin. Journal ofComparative Neurology, 342(2): 206–220, 1994. doi: 10.1002/cne.903420205.

[316] Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, and Abhinav Gupta. The curious robot: Learning visual representations via physical interactions. In European Conference on Computer Vision (ECCV), 2016.

[317] David Pitcher and Leslie G. Ungerleider. Evidence for a third visual pathway specialized for social perception. Trends in Cognitive Sciences, 25(2):100–110, 2021. doi: 10.1016/j.tics.2020.11.006.

[318] Alexandre Pouget and Terrence J. Sejnowski. Spatial transformations in the parietal cortex using basis functions. Journal ofCognitive Neuroscience, 9(2):222–237, 1997.

[319] Claude Prablanc and Olivier Martin. Automatic control during hand reaching at undetected two-dimensional target displacements. Journal ofNeurophysiology, 67(2):455–469, 1992. doi: 10.1152/jn.1992.67.2.455.

[320] Gill A. Pratt and Matthew M. Williamson. Series elastic actuators. In Proceedings ofthe IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), volume 1, pages 399–406, 1995.

[321] J. Andrew Pruszynski and Stephen H. Scott. Optimal feedback control and the long-latency stretch response. Experimental Brain Research, 218(3):341–359, 2012.

[322] Michael Przystupa, Hongyao Tang, Martin Jagersand, Santiago Miret, Mariano Phielipp, Matthew E. Taylor, and Glen Berseth. Eficient morphology-aware policy transfer to new embodiments. arXiv preprint arXiv:2508.03660, 2025. Reinforcement Learning Conference 2025.

[323] Zenon W. Pylyshyn and Ron W. Storm. Tracking multiple independent targets: Evidence for a parallel tracking mechanism. Spatial Vision, 3(3):179–197, 1988. doi: 10.1163/156856888X00122.

[324] Delin Qu, Haoming Song, Qizhi Chen, et al. SpatialVLA: Exploring spatial representations for visual-languageaction model. arXiv preprint arXiv:2501.15830, 2025.

[325] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.

[326] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, volume 20, 2007.

[327] Dhakshin S. Ramanathan, Tanuj Gulati, and Karunesh Ganguly. Sleep-dependent reactivation of ensembles in motor cortex promotes skill consolidation. PLoS Biology, 13(9):e1002263, 2015.

[328] John H. Reynolds and David J. Heeger. The normalization model of attention. Neuron, 61(2):168–185, 2009. doi: 10.1016/j.neuron.2009.01.002.

[329] José J. F. Ribas-Fernandes, Alec Solway, Carlos Diuk, Joseph T. McGuire, Andrew G. Barto, Yael Niv, and Matthew M. Botvinick. A neural signature of hierarchical reinforcement learning. Neuron, 71(2):370–379, 2011.

[330] C. Nicholas Riddle, Steve A. Edgley, and Stuart N. Baker. Direct and indirect connections with upper limb motoneurons from the primate reticulospinal tract. The Journal ofNeuroscience, 29(15):4993–4999, 2009. doi: 10.1523/JNEUROSCI.3720-08.2009.

[331] Giacomo Rizzolatti and Massimo Matelli. Two diferent streams form the dorsal visual system: anatomy and functions. Experimental Brain Research, 153:146–157, 2003.

[332] Stéphane Ross, Geofrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, pages 627–635, 2011.

[333] Patrick T. Sadtler, Kristin M. Quick, Matthew D. Golub, Steven M. Chase, Stephen I. Ryu, Elizabeth C. Tyler-Kabara, Byron M. Yu, and Aaron P. Batista. Neural constraints on learning. Nature, 512(7515):423–426, 2014. doi: 10.1038/nature13665.

[334] Katsuyuki Sakai, Katsuya Kitaguchi, and Okihide Hikosaka. Chunking during human visuomotor sequence learning. Experimental Brain Research, 152(2):229–242, 2003.

[335] Hideo Sakata, Masato Taira, Akira Murata, and Seigo Mine. Neural mechanisms of visual guidance of hand action in the parietal cortex of the monkey. Cerebral Cortex, 5(5):429–438, 1995.

[336] Anupama Sathyamurthy, Arnab Barik, Courtney I. Dobrott, Kaya J. E. Matson, Stefan Stoica, Randall Pursley, Alexander T. Chesler, and Ariel J. Levine. Cerebellospinal neurons regulate motor performance and motor learning. Cell Reports, 31(6):107595, 2020. doi: 10.1016/j.celrep.2020.107595.

[337] Britton A. Sauerbrei, Jian-Zhong Guo, Jeremy D. Cohen, Matteo Mischiati, Wendy Guo, Mayank Kabra, Naku Verma, Brett Mensh, Kristin Branson, and Adam W. Hantman. Cortical pattern generation during dexterous movement is input-driven. Nature, 577:386–391, 2020.

[338] Stefan Schafelhofer and Hansjörg Scherberger. Object vision to hand action in macaque parietal, premotor, and motor cortices. eLife, 5:e15278, 2016.

[339] Thomas Schenk and Robert D. McIntosh. Do we have independent visual streams for perception and action? Cognitive Neuroscience, 1(1):52–62, 2010. doi: 10.1080/17588920903388950.

[340] Wolfram Schultz, Peter Dayan, and P. Read Montague. A neural substrate of prediction and reward. Science, 275 (5306):1593–1599, 1997.

[341] Florian Seligmann, Emiliyan Gospodinov, Enes Ulas Dincer, and Gerhard Neumann. Vla-fail: Eficient task failure detection for finetuned vision-language-action models. arXiv preprint arXiv:2606.21386, 2026.

[342] Seyed Kamyar Seyed Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, and Igor Mordatch. Self-improving embodied foundation models. In Advances in Neural Information Processing Systems 38, NeurIPS 2025, page 124887–124922. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025. doi: 10.52202/085713-3762. URL http://dx.doi.org/10.52202/085713-3762.

[343] Carmelo Sferrazza, Dun-Ming Huang, Fangchen Liu, Jongmin Lee, and Pieter Abbeel. Body transformer: Leveraging robot embodiment for policy learning. arXiv preprint arXiv:2408.06316, 2024.

[344] Rutav Shah, Shuijing Liu, Qi Wang, Zhenyu Jiang, Sateesh Kumar, Mingyo Seo, Roberto Martín-Martín, and Yuke Zhu. Mimicdroid: In-context learning for humanoid robot manipulation from human play videos. arXiv preprint arXiv:2509.09769, 2025.

[345] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations (ICLR), 2017.

[346] Yide Shentu, Philipp Wu, Aravind Rajeswaran, and Pieter Abbeel. From LLMs to actions: Latent codes as bridges in hierarchical robot control. arXiv preprint arXiv:2405.04798, 2024.

[347] Charles S. Sherrington. The Integrative Action ofthe Nervous System. Yale University Press, New Haven, CT, 1906.

[348] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025.

[349] Modi Shi, Li Chen, Jin Chen, Yuxiang Lu, Chiming Liu, Guanghui Ren, Ping Luo, Di Huang, Maoqing Yao, and Hongyang Li. Is diversity all you need for scalable robotic manipulation? arXiv preprint arXiv:2507.06219, 2025.

[350] Munetaka Shidara and Barry J. Richmond. Anterior cingulate: Single neuronal signals related to degree of reward expectancy. Science, 296(5573):1709–1711, 2002.

[351] M. L. Shik, F. V. Severin, and G. N. Orlovsky. Control of walking and running by means of electrical stimulation of the mid-brain. Biofizika, 11:659–666, 1966.

[352] Keisetsu Shima and Jun Tanji. Neuronal activity in the supplementary and presupplementary motor areas for temporal organization of multiple movements. Journal of Neurophysiology, 84(4):2148–2160, 2000.

[353] Lior Shmuelof, John W. Krakauer, and Pietro Mazzoni. How is a motor skill learned? Change and invariance at the levels of task success and trajectory control. Journal ofNeurophysiology, 108(2):578–594, 2012. doi: 10.1152/jn.00856.2011.

[354] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. SmolVLA: A vision-language-action model for afordable and eficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[355] F. Shutoh, M. Ohki, H. Kitazawa, S. Itohara, and S. Nagao. Memory trace of motor learning shifts transiently from cerebellar cortex to nuclei for consolidation. Neuroscience, 139(2):767–777, 2006. doi: 10.1016/j.neuroscience. 2005.12.035.

[356] Ruiyang Si, Jianxin Bi, Shunyu Yang, Rui Ni, Wenbo Huang, Qiang Wang, Shulong Jiang, Duomin Wang, Xiuyu Li, Haiwen Feng, Zhen Dong, and Daquan Zhou. Fewer tokens, better action: GPT-6 Astra robot agents with 14% higher success rate but 65% fewer tokens. arXiv preprint arXiv:2610.01939, 2026.

[357] Bruno Siciliano, Lorenzo Sciavicco, Luigi Villani, and Giuseppe Oriolo. Robotics: Modelling, Planning and Control. Advanced Textbooks in Control and Signal Processing. Springer, 2009.

[358] N. Tatiana Silva, Jorge Ramírez-Buriticá, Dominique L. Pritchett, and Megan R. Carey. Climbing fibers provide essential instructive signals for associative learning. Nature Neuroscience, 27(5):940–951, April 2024. ISSN 1546- 1726. doi: 10.1038/s41593-024-01594-7. URL http://dx.doi.org/10.1038/s41593-024-01594-7.

[359] Oriane Siméoni, Huy V. Vo, et al. DINOv3. arXiv preprint arXiv:2508.10104, Meta AI Research, 2025.

[360] Amarjot Singh, Tanmay R. Pancholi, Jainam Kothari, Shrirang Mahajan, Ketan Bansal, Zackory Erickson, Giuseppe Loianno, Alexandre M. Bayen, Jef Schneider, and Vince Nakayama. Continual field-adaptive models (CFAMs) for post-deployment physical AI. arXiv preprint arXiv:2609.04552, 2026.

[361] Marc A. Sommer and Robert H. Wurtz. Influence of the thalamus on spatial visual processing in frontal cortex. Nature, 444:374–377, 2006.

[362] Dongwon Son, Florian Shkurti, Jason Lee, Naman Shah, Beomjoon Kim, and Dieter Fox. TAM: Torque adaptation module for robust motion transfer in manipulation. arXiv preprint arXiv:2606.06218, 2026.

[363] Constantino Sotelo. Cellular and genetic regulation of the development of the cerebellar system. Progress in Neurobiology, 72(5):295–339, 2004.

[364] Peter L. Strick, Richard P. Dum, and Jean-Alban Rathelot. The cortical motor areas and the emergence of motor skills: A neuroanatomical perspective. Annual Review ofNeuroscience, 44:425–447, 2021.

[365] Lin Sun, Zhiwei Guan, Conglin Wang, Zihong Chen, Jianhai Yu, Zongsheng Li, Boyong He, Tao Sun, Jiale Cao, and Lige Liu. UniFS: Unified fast-to-slow hierarchical architecture for vision-language-action models. arXiv preprint arXiv:2606.22794, 2026.

[366] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024.

[367] Subramanian Sundaram, Petr Kellnhofer, Yunzhu Li, Jun-Yan Zhu, Antonio Torralba, and Wojciech Matusik. Learning the signatures of the human grasp using a scalable tactile glove. Nature, 569(7758):698–702, 2019.

[368] Suvrathan, Payne, and Raymond. Timing rules for synaptic plasticity matched to behavioral function. Neuron, 92 (5):959–967, 2016.

[369] Kei Suzuki, Jing Liu, Ye Wang, Chiori Hori, Matthew Brand, Diego Romeres, and Toshiaki Koike-Akino. Embedding morphology into transformers for cross-robot policy learning. arXiv preprint arXiv:2603.00182, 2026.

[370] Jiaming Tang, Yufei Sun, Yilong Zhao, Shang Yang, Yujun Lin, Zhuoyang Zhang, James Hou, Yao Lu, Zhijian Liu, and Song Han. VLASH: Real-time VLAs via future-state-aware asynchronous inference. arXiv preprint arXiv:2512.01031, 2025.

[371] Jun Tanji and Keisetsu Shima. Role for supplementary motor area cells in planning several movements ahead. Nature, 371(6496):413–416, 1994.

[372] Jordan A. Taylor, John W. Krakauer, and Richard B. Ivry. Explicit and implicit contributions to learning in a sensorimotor adaptation task. Journal ofNeuroscience, 34(8):3023–3032, 2014.

[373] Yaroslav Tenzer, Leif P. Jentoft, and Robert D. Howe. The feel of MEMS barometers: Inexpensive and easily customized tactile array sensors. IEEE Robotics & Automation Magazine, 21(3):89–95, 2014. doi: 10.1109/MRA.2014.2310152.

[374] Kurt A. Thoroughman and Reza Shadmehr. Electromyographic correlates of learning an internal model of reaching movements. Journal ofNeuroscience, 19(19):8573–8588, 1999. doi: 10.1523/JNEUROSCI.19-19-08573.1999.

[375] Simon Thorpe, Denis Fize, and Catherine Marlot. Speed of processing in the human visual system. Nature, 381: 520–522, 1996. doi: 10.1038/381520a0.

[376] Shuai Tian, Yupeng Zheng, Yuhang Zheng, Songen Gu, Yujie Zang, Yuxing Qin, Weize Li, Haoran Li, Wenchao Ding, and Dongbin Zhao. VT-WAM: Visual-tactile world action model for contact-rich manipulation, 2026. URL https://arxiv.org/abs/2607.02503.

[377] Alexandre Tiriac and Mark S. Blumberg. Gating of reaference in the external cuneate nucleus during self-generated movements in wake but not sleep. eLife, 5:e18749, 2016.

[378] B. E. Tomlinson and D. Irving. The numbers of limb motor neurons in the human lumbosacral cord throughout life. Journal ofthe Neurological Sciences, 34(2):213–219, 1977.

[379] Tito Pradhono Tomo, Alexander Schmitz, Wai Keat Wong, Harris Kristanto, Sophon Somlor, Jinsun Hwang, Lorenzo Jamone, and Shigeki Sugano. A new silicone structure for uSkin — a soft, distributed, digital 3-axis skin sensor and its integration on the humanoid robot iCub. IEEE Robotics and Automation Letters, 3(3):2584–2591, 2018. doi: 10.1109/LRA.2018.2812915.

[380] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: Masked autoencoders are data-eficient learners for self-supervised video pre-training. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[381] Marcel Torne, Karl Pertsch, Homer Walke, et al. MEM: Multi-scale embodied memory for vision language action models. arXiv preprint arXiv:2603.03596, 2026.

[382] Stefan Treue and John H. R. Maunsell. Attentional modulation of visual motion processing in cortical areas MT and MST. Nature, 382:539–541, 1996. doi: 10.1038/382539a0

[383] Nikolaos Tsagkas, Andreas Sochopoulos, Duolikun Danier, Chris Xiaoxuan Lu, and Oisin Mac Aodha. When pre-trained visual representations fall short: Limitations in visuo-motor robot learning. arXiv preprint arXiv:2502.03270, 2025.

[384] Jonathan S Tsay, Hyosub E Kim, Samuel D McDougle, Jordan A Taylor, Adrian Haith, Guy Avraham, John W Krakauer, Anne GE Collins, and Richard B Ivry. Fundamental processes in sensorimotor learning: Reasoning, refinement, and retrieval. eLife, 13, August 2024. ISSN 2050-084X. doi: 10.7554/elife.91839. URL http://dx.doi.org/10.7554/eLife.91839.

[385] Michael Tschannen, Alexey Gritsenko, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[386] William Turner, Charlie Sexton, and Hinze Hogendoorn. Neural mechanisms of visual motion extrapolation. Neuroscience & Biobehavioral Reviews, 156:105484, 2024. doi: 10.1016/j.neubiorev.2023.105484.

[387] Maria Alessandra Umiltà, L. Escola, I. Intskirveli, F. Grammont, M. Rochat, F. Caruana, A. Jezzini, Vittorio Gallese, and Giacomo Rizzolatti. When pliers become fingers in the monkey motor system. Proceedings ofthe National Academy ofSciences (PNAS), 105(6):2209–2213, 2008.

[388] Leslie G. Ungerleider and Mortimer Mishkin. Two cortical visual systems. In David J. Ingle, Melvyn A. Goodale, and Richard J. W. Mansfield, editors, Analysis of Visual Behavior, pages 549–586. MIT Press, Cambridge, MA, 1982.

[389] Å. B. Vallbo and R. S. Johansson. Properties of cutaneous mechanoreceptors in the human hand related to touch sensation. Human Neurobiology, 3(1):3–14, 1984.

[390] Vitalis Vosylius and Edward Johns. Instant policy: In-context imitation learning via graph difusion. In International Conference on Learning Representations (ICLR), 2025. arXiv:2411.12633.

[391] Mark J. Wagner, Tony Hyun Kim, Joan Savall, Mark J. Schnitzer, and Liqun Luo. Cerebellar granule cells encode the expectation of reward. Nature, 544(7648):96–100, 2017.

[392] Matthew P. Walker, Tifany Brakefield, Alexandra Morgan, J. Allan Hobson, and Robert Stickgold. Practice with sleep makes perfect: Sleep-dependent motor skill learning. Neuron, 35(1):205–211, 2002. doi: 10.1016/ S0896-6273(02)00746-8.

[393] Jonathan D. Wallis, Kathleen C. Anderson, and Earl K. Miller. Single neurons in prefrontal cortex encode abstract rules. Nature, 411(6840):953–956, 2001.

[394] Joy T. Walter and Kamran Khodakhah. The linear computational algorithm of cerebellar Purkinje cells. Journal ofNeuroscience, 26(50):12861–12872, 2006. doi: 10.1523/JNEUROSCI.4507-05.2006.

[395] Yang Wan, Xihang Yue, Zhirui Liu, Ziyuan Chu, Shuxun Wang, Yuhan Chen, Xiaonan Jiang, Xukun Zhu, Yubo Dong, and Linchao Zhu. No free checker: A survey of verifiers for robot policies. arXiv preprint arXiv:2609.09250, 2026.

[396] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

[397] Jiaming Wang. What stops recursive self-improvement in robotics? Lessons from 123 rounds of agentic skill discovery. arXiv preprint arXiv:2609.31760, 2026.

[398] Lirui Wang, Xinlei Chen, Jialiang Zhao, and Kaiming He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[399] Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, Mike Zheng Shou, Xuanjing Huang, Xipeng Qiu, and Yu-Gang Jiang. World action models: The next frontier in embodied AI. arXiv preprint arXiv:2605.12090, 2026.

[400] Xinyi Wang, Bangzheng Li, Hanyu Wang, et al. SEAR: Self-evolving agentic robotics. arXiv preprint arXiv:2610.18792, 2026.

[401] Yi Wang, Xinchen Li, Pengwei Xie, et al. Learning while deploying: Fleet-scale reinforcement learning for generalist robot policies. arXiv preprint arXiv:2605.00416, 2026.

[402] Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren. AdaJEPA: An adaptive latent world model. arXiv preprint arXiv:2606.32026, 2026.

[403] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025.

[404] Zhenan Wang, Yanzhe Wang, Meixuan Ren, Peng Li, Yang Liu, Yifei Nie, Limin Long, Yun Ye, Xiaofeng Wang, Zhen Zhu, and Huixu Dong. TacMamba: A tactile history compression adapter bridging fast reflexes and slow VLA reasoning. arXiv preprint arXiv:2603.01700, 2026.

[405] Benjamin Ward-Cherrier, Nicholas Pestell, Luke Cramphorn, Benjamin Winstone, Maria Elena Giannaccini, Jonathan Rossiter, and Nathan F. Lepora. The TacTip family: Soft optical tactile sensors with 3D-printed biomimetic morphologies. Soft Robotics, 5(2):216–227, 2018. doi: 10.1089/soro.2017.0052.

[406] Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point trajectory modeling for policy learning. In Robotics: Science and Systems (RSS), 2024.

[407] Zimo Wen, Yijin Chen, Yuxuan Cao, et al. RoboRSI: Stable, eficient, and reusable robot self-evolution in complex real-world environments. arXiv preprint arXiv:2610.12424, 2026.

[408] Patrick M. Wensing, Albert Wang, Sangok Seok, David Otten, Jefrey Lang, and Sangbae Kim. Proprioceptive actuator design in the MIT Cheetah: Impact mitigation and high-bandwidth physical interaction for dynamic legged robots. IEEE Transactions on Robotics, 33(3):509–522, 2017. doi: 10.1109/TRO.2016.2640183.

[409] Nicholas Wettels, Veronica J. Santos, Roland S. Johansson, and Gerald E. Loeb. Biomimetic tactile sensor array. Advanced Robotics, 22(8):829–849, 2008. doi: 10.1163/156855308X314533.

[410] Daniel E. Whitney. Quasi-static assembly of compliantly supported rigid parts. Journal ofDynamic Systems, Measurement, and Control, 104(1):65–77, 1982. doi: 10.1115/1.3149634.

[411] Ralf D. Wimmer, L. Ian Schmitt, Thomas J. Davidson, Miho Nakajima, Karl Deisseroth, and Michael M. Halassa. Thalamic control of sensory selection in divided attention. Nature, 526(7575):705–709, 2015. doi: 10.1038/nature15398.

[412] Daniel M. Wolpert, Zoubin Ghahramani, and Michael I. Jordan. An internal model for sensorimotor integration. Science, 269(5232):1880–1882, 1995. doi: 10.1126/science.7569931.

[413] Daniel M. Wolpert, R. Chris Miall, and Mitsuo Kawato. Internal models in the cerebellum. Trends in Cognitive Sciences, 2(9):338–347, 1998.

[414] Howard G. Wu, Yohsuke R. Miyamoto, Luis Nicolas Gonzalez Castro, Bence P. Ölveczky, and Maurice A. Smith. Temporal structure of motor variability is dynamically regulated and predicts motor learning ability. Nature Neuroscience, 17(2):312–321, 2014.

[415] Robert H. Wurtz. Neuronal mechanisms of visual stability. Vision Research, 48(20):2070–2089, 2008.

[416] Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi "Jim" Fan, Guanya Shi, and Yuke Zhu. Self-improving vision-language-action models with data generation via residual RL. arXiv preprint arXiv:2511.00091, 2025. ICLR 2026.

[417] Yang Xie, Peiyao Hu, Junru Li, Jingwen Chen, Weibin Song, Xiao-Jing Wang, Tianming Yang, Stanislas Dehaene, Shiming Tang, Bin Min, and Liping Wang. Geometry of sequence working memory in macaque prefrontal cortex. Science, 375(6581):632–639, 2022. doi: 10.1126/science.abm0204.

[418] Yichen Xie, Yixiao Wang, Shuqi Zhao, Cheng-En Wu, Masayoshi Tomizuka, Jianwen Xie, and Hao-Shu Fang. Multi-camera view scaling for data-eficient robot imitation learning. arXiv preprint arXiv:2604.00557, 2026.

[419] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. SimMIM: A simple framework for masked image modeling. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[420] Bingxin Xu, Yuzhang Shang, and Emilio Ferrara. Don’t drop the BATON: Long-horizon robot manipulation via agentic subtask exploration and transition-aware memory. arXiv preprint arXiv:2608.16889, 2026.

[421] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. In Conference on Robot Learning (CoRL), 2024.

[422] Tonghui Xu, Xinzhu Yu, Andrew J. Perlik, Willie F. Tobin, Jonathan A. Zweig, Kelly Tennant, Theresa Jones, and Yi Zuo. Rapid formation and selective stabilization of synapses for enduring motor memories. Nature, 462 (7275):915–919, 2009.

[423] Sho Yagishita, Akiko Hayashi-Takagi, Graham C. R. Ellis-Davies, Hidetoshi Urakubo, Shin Ishii, and Haruo Kasai. A critical time window for dopamine actions on the structural plasticity of dendritic spines. Science, 345 (6204):1616–1620, 2014. doi: 10.1126/science.1255514.

[424] Akihiko Yamaguchi and Christopher G. Atkeson. Recent progress in tactile sensing and sensors for robotic manipulation: Can we turn tactile sensing into vision? Advanced Robotics, 33(14):661–673, 2019. doi: 10.1080/01691864.2019.1632222.

[425] Fengyu Yang, Chao Feng, Ziyang Wang, Tingbo Huang, Andrew Owens, et al. Binding touch to everything: Learning unified multimodal tactile representations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[426] Guang Yang, Feng Pan, and Wen-Biao Gan. Stably maintained dendritic spines are associated with lifelong memories. Nature, 462(7275):920–924, 2009.

[427] Guang Yang, Cora Sau Wan Lai, Joseph Cichon, Lei Ma, Wei Li, and Wen-Biao Gan. Sleep promotes branch-specific formation of dendritic spines after learning. Science, 344(6188):1173–1178, 2014.

[428] Yan Yang and Stephen G. Lisberger. Purkinje-cell plasticity and cerebellar motor learning are graded by complex-spike duration. Nature, 510(7506):529–532, 2014.

[429] Alfred L. Yarbus. Eye Movements and Vision. Plenum Press, New York, 1967.

[430] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024.

[431] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Jim Fan, and Joel Jang. World action models are zero-shot policies (DreamZero), 2026. URL https://arxiv.org/abs/2602.15922.

[432] Wanghao Ye, Aarosh Das, Sihan Chen, Yiting Wang, Bowei Tian, Guoheng Sun, Shwai He, Zheyu Shen, Ziyao Wang, Yexiao He, et al. TacGen: Touch is a necessary dimension of physical-world representation—addressing tactile data scarcity with scalable vision-to-touch alignment and generation. arXiv preprint arXiv:2606.29173, 2026.

[433] Henry H. Yin, Shweta Prasad Mulcare, Monica R. F. Hilário, et al. Dynamic reorganization of striatal circuits during the acquisition and consolidation of a skill. Nature Neuroscience, 12(3):333–341, 2009.

[434] Hanna Yousef, Mehdi Boukallel, and Kaspar Althoefer. Tactile sensing for dexterous in-hand manipulation in robotics—a review. Sensors and Actuators A: Physical, 167(2):171–187, 2011. doi: 10.1016/j.sna.2011.02.038.

[435] Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, et al. ForceVLA: Enhancing VLA models with a force-aware MoE for contact-rich manipulation. arXiv preprint arXiv:2505.22159, 2025.

[436] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. In IEEE/CVF Conference on Compute Vision and Pattern Recognition (CVPR), 2024.

[437] Kelin Yu, Haode Zhang, Harish Ravichandar, Yunhai Han, and Ruohan Gao. OmniTacTune: Policy-agnostic real-world RL for tactile residual adaptation of visual policies. arXiv preprint arXiv:2607.03723, 2026.

[438] Zhaoshu Yu, Bo Wang, Pengpeng Zeng, et al. A survey on eficient vision-language-action models. arXiv preprint arXiv:2510.24795, 2025.

[439] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[440] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. RoboPoint: A vision-language model for spatial afordance prediction for robotics. In Conference on Robot Learning, 2024.

[441] Wenzhen Yuan, Siyuan Dong, and Edward H. Adelson. GelSight: High-resolution robot tactile sensors for estimating geometry and force. Sensors, 17(12):2762, 2017. doi: 10.3390/s17122762.

[442] Felix E. Zajac. Muscle and tendon: Properties, models, scaling, and application to biomechanics and motor control. Critical Reviews in Biomedical Engineering, 17(4):359–411, 1989.

[443] Yujie Zang, Yuhang Zheng, Xian Nie, Yupeng Zheng, Shuai Tian, Songen Gu, Chen Gao, Zining Wang, Shuicheng Yan, and Wenchao Ding. TacForeSight: Force-guided tactile world model for contact-rich manipulation. arXiv preprint arXiv:2606.11184, 2026.

[444] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. In Conference on Robot Learning, 2024.

[445] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. arXiv:2303.15343.

[446] Tonghe Zhang, Chao Yu, Sichang Su, and Yu Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094, 2025. NeurIPS 2025.

[447] Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, Haotian Liang, Zhiyang Dou, Yue Chen, and Tianxing Chen. An unexpected robot policy: Early evaluations of GPT-6 Astra on RoboDojo and beyond. arXiv preprint arXiv:2609.24170, 2026.

[448] Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Yi Nie, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, Jiaji Rao, Zhengru Fang, Changxu Wei, Yu Wang, Wenbo Ding, and Chao Yu. Harness VLA: Steering frozen VLAs into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

[449] Zhanguang Zhang, Zhiyuan Li, Behnam Rahmati, Rui Heng Yang, Yintao Ma, Amir Rasouli, Sajjad Pakdamansavoji, Yangzheng Wu, Lingfeng Zhang, Tongtong Cao, Feng Wen, Xinyu Wang, Xingyue Quan, and Yingxue Zhang. Do world action models generalize better than VLAs? A robustness study. arXiv preprint arXiv:2603.22078, 2026.

[450] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[451] Zihang Zhao, Wanlin Li, Yuyang Li, Tengyu Liu, Boren Li, Meng Wang, Kai Du, Hangxin Liu, Yixin Zhu, Qining Wang, Kaspar Althoefer, and Song-Chun Zhu. Embedding high-resolution touch across robotic hands enables adaptive human-like grasping. Nature Machine Intelligence, 7(6):889–900, 2025. doi: 10.1038/ s42256-025-01053-3.

[452] Zan Zheng, Jin Huang, Chao Wang, Xi Tan, Jie Xiao, Fang Zhang, and Lulu Liu. Bridging slow reasoning and fast acting: A dual-system survey of vision-language-action architectures. SSRN preprint 7139866, https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=7139866, 2026.

[453] Hai Zhou, Fan Yang, Zeyi Chen, et al. Who improves the robot? Toward physical recursive self-improvement. arXiv preprint arXiv:2610.18842, 2026.

[454] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action difusion for pretraining on large robotic datasets. In Robotics: Science and Systems (RSS), 2025. doi: 10.15607/RSS.2025.XXI.015.

[455] Fangqi Zhu, Zhengyang Yan, Zicong Hong, Quanxin Shou, Xiao Ma, and Song Guo. WMPO: World model-based policy optimization for vision-language-action models. arXiv preprint arXiv:2511.09515, 2025.

[456] Manfred Zimmermann. Neurophysiology of sensory systems. In Robert F. Schmidt, editor, Fundamentals of Sensory Physiology, pages 68–116. Springer, Berlin, 1986.
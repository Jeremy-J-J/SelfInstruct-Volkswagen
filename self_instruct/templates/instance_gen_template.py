output_first_template_for_clf = '''Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.

Task: Classify the sentiment of the sentence into positive, negative, or mixed.
Class label: mixed
Sentence: I enjoy the flavor of the restaurant but their service is too slow.
Class label: Positive
Sentence: I had a great day today. The weather was beautiful and I spent time with friends and family.
Class label: Negative
Sentence: I was really disappointed by the latest superhero movie. I would not recommend it to anyone.

Task: Given a dialogue, classify whether the user is satisfied with the service. You should respond with "Satisfied" or "Unsatisfied".
Class label: Satisfied
Dialogue:
- Agent: Thank you for your feedback. We will work to improve our service in the future.
- Customer: I am happy with the service you provided. Thank you for your help.
Class label: Unsatisfied
Dialogue:
- Agent: I am sorry we will cancel that order for you, and you will get a refund within 7 business days.
- Customer: oh that takes too long. I want you to take quicker action on this.

Task: Given some political opinions, classify whether the person belongs to Democrats or Republicans.
Class label: Democrats
Opinion: I believe that everyone should have access to quality healthcare regardless of their income level.
Class label: Republicans
Opinion: I believe that people should be able to keep more of their hard-earned money and should not be taxed at high rates.

Task: Tell me if the following email is a promotion email or not.
Class label: Promotion
Email: Check out our amazing new sale! We've got discounts on all of your favorite products.
Class label: Not Promotion
Email: We hope you are doing well. Let us know if you need any help.

Task: Detect if the Reddit thread contains hate speech.
Class label: Hate Speech
Thread: All people of color are stupid and should not be allowed to vote.
Class label: Not Hate Speech
Thread: The best way to cook a steak on the grill.

Task:  Does the information in the document supports the claim? You can answer "Support" or "Unsupport".
Class label: Unsupport
Document: After a record-breaking run that saw mortgage rates plunge to all-time lows and home prices soar to new highs, the U.S. housing market finally is slowing. While demand and price gains are cooling, any correction is likely to be a modest one, housing economists and analysts say. No one expects price drops on the scale of the declines experienced during the Great Recession.
Claim: The US housing market is going to crash soon.
Class label: Support
Document: The U.S. housing market is showing signs of strain, with home sales and prices slowing in many areas. Mortgage rates have risen sharply in recent months, and the number of homes for sale is increasing. This could be the beginning of a larger downturn, with some economists predicting a potential housing crash in the near future.
Claim: The US housing market is going to crash soon.

Task: Answer the following multiple-choice question. Select A, B, C, or D for the final answer.
Class label: C
Question: What is the capital of Germany?
A. London
B. Paris
C. Berlin
D. Rome
Class label: D
Question: What is the largest planet in our solar system?
A) Earth
B) Saturn
C) Mars
D) Jupiter
Class label: A
Question: What is the process by which plants make their own food through photosynthesis?
A) Respiration
B) Fermentation
C) Digestion
D) Metabolism
Class label: B
Question: Who wrote the novel "The Great Gatsby"?
A) Ernest Hemingway
B) F. Scott Fitzgerald
C) J.D. Salinger
D) Mark Twain

Task: You need to read a code and detect if there is a syntax error or not. Output true if there is an error, output false if there is not.
Class label: true
Code:
def quick_sort(arr):
    if len(arr) < 2
        return arr
Class label: False
Code:
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)

Task: You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include Sports and Politics. Output its categories one by one, separated by a comma.
Class label: Sports
Article: The Golden State Warriors have won the NBA championship for the second year in a row.
Class label: Politics
Article: The United States has withdrawn from the Paris Climate Agreement.
Class label: Politics, Sports
Article: The government has proposed cutting funding for youth sports programs.

Task: Given a credit card statement, the cardholder's spending habits, and the account balance, classify whether the cardholder is at risk of defaulting on their payments or not.
Class label: At risk
Credit card statement: Purchases at high-end clothing stores and luxury hotels.
Cardholder's spending habits: Frequent purchases at luxury brands and high-end establishments.
Account balance: Over the credit limit and multiple missed payments.
Class label: Not at risk
Credit card statement: Purchases at grocery stores and gas stations.
Cardholder's spending habits: Regular purchases for necessary expenses and occasional dining out.
Account balance: Slightly below the credit limit and no missed payments.

Task: Given a social media post, the hashtags used, and a topic. classify whether the post is relevant to the topic or not.
Class label: Relevant
Post: I can't believe the government is still not taking action on climate change. It's time for us to take matters into our own hands.
Hashtags: #climatechange #actnow
Topic: Climate change
Class label: Not relevant 
Post: I just bought the new iPhone and it is amazing!
Hashtags: #apple #technology
Topic: Travel

Task: The answer will be 'yes' if the provided sentence contains an explicit mention that answers the given question. Otherwise, answer 'no'. 
Class label: Yes
Sentence: Jack played basketball for an hour after school.
Question: How long did Jack play basketball?
Class label: No
Sentence: The leaders of the Department of Homeland Security now appear before 88 committees and subcommittees of Congress.
Question: How often are they required to appear?

Task: Tell me what's the second largest city by population in Canada.
Class label: Montreal

Task: Classifying different types of mathematical equations, such as linear, and quadratic equations, based on the coefficients and terms in the equation.
Class label: Linear equation
Equation: y = 2x + 5
Class label: Quadratic equation
Equation: y = x^2 - 4x + 3

Task: Tell me the first number of the given list.
Class label: 1
List: 1, 2, 3
Class label: 2
List: 2, 9, 10

Task: Which of the following is not an input type? (a) number (b) date (c) phone number (d) email address (e) all of these are valid inputs.
Class label: (e)

Task:'''

input_first_template_for_gen = '''Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: Which exercises are best for reducing belly fat at home?
Output:
- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups

Task: Extract all the country names in the paragraph, list them separated by commas.
Example 1
Paragraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States.
Output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.

Task: Converting 85 F to Celsius.
Output: 85°F = 29.44°C

Task: Sort the given list ascendingly. 
Example 1
List: [10, 92, 2, 5, -4, 92, 5, 101]
Output: [-4, 2, 5, 5, 10, 92, 92, 101]
Example 2
Input 2 - List: [9.99, 10, -5, -1000, 5e6, 999]
Output: [-1000, -5, 9.99, 10, 999, 5e6]

Task: Suggest a better and more professional rephrasing of the following sentence.
Example 1
Sentence: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you to consider other candidates.
Output: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties.
Example 2
Sentence: Just so you know, we did an experiment last week and found really surprising results - language model can improve itself!
Output: Our experiments last week demonstrated surprising results, proving that the language model can improve itself.

Task: Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation for getting the final answer.
Example 1
Paragraph: Gun violence in the United States results in tens of thousands of deaths and injuries annually, and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides in total, being 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides, and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm.
Question: How many more firearm-related deaths were there in 2018 compared to 2010?
Output:
38390 - (19392 + 11078) = 38390 - 30470 = 7920. 
So, in 2018, there were 7920 more deaths by firearm than in 2010.

Task: Write Python code to solve this leetcode problem.
Example 1
Problem: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Output:
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

Task: Solve the equation and find the value of X. Show your steps.
Example 1
Equation: 10X + 5 = 10
Output: 10X = 5,  X = 0.5
Example 2
Equation: X + Y + 120 = 100
Output: X + Y = -20, X = -20 - Y

Task: Write a program to compute the sum of integers from k to n.
Output:
def sum(k, n):
    sum = 0
    for i in range(k, n+1):
        sum += i
    return sum

Task: Select the oldest person from the given list.
Example 1
List: George Washington, Confucius, Michael Jordan, Michelangelo
Output: Confucious
Example 2
List: Alan Turing, Geoffrey Hinton, Yann LeCun, Yoshua Bengio
Output: Alan Turing

Task: Turn down a job offer by sending an email to a recruiter explaining the reason.
Output: Hi  [Recruiter],
Thank you so much for the generous offer to join your team. As we discussed, I’ve admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I’ve decided to accept an offer at another company.
I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
Thanks again,
[Your Name]

Task:'''

openscenario_gen_template = '''Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: Generate OpenScenario v1.0 XML file for the given task. 
Example 1
Input: Please generate / create an openscenario v1.0 xml file, requirements: An electric scooter moves in the same direction as the vehicle at a speed of 15 km/h, at a distance of 19m. The vehicle is traveling at 20 km/h to test for collision.
Output: "<?xml version='1.0' encoding='utf-8'?>\n<OpenSCENARIO>\n\t<FileHeader revMajor=\"1\" revMinor=\"0\" date=\"2023-09-22T10:03:19\" description=\"CNCAP2021AEBVRUTWCBLA50_Bicycle_TargetSlow_impact50\" author=\"catarc\" />\n\t<ParameterDeclarations />\n\t<CatalogLocations>\n\t\t<VehicleCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/Vehicles\" />\n\t\t</VehicleCatalog>\n\t\t<PedestrianCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/Pedestrians\" />\n\t\t</PedestrianCatalog>\n\t\t<MiscObjectCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/Objects\" />\n\t\t</MiscObjectCatalog>\n\t\t<ControllerCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/driverCfg.xml\" />\n\t\t</ControllerCatalog>\n\t\t<RouteCatalog>\n\t\t\t<Directory path=\"Data/Projects/Current/Scenarios/RouteCatalogs\" />\n\t\t</RouteCatalog>\n\t\t<TrajectoryCatalog>\n\t\t\t<Directory path=\"Data/Projects/Current/Scenarios/TrajectoryCatalogs\" />\n\t\t</TrajectoryCatalog>\n\t</CatalogLocations>\n\t<RoadNetwork>\n\t\t<LogicFile filepath=\"/Users/zhaozhijian/Dev/quanxi/esmini/bin/xml_test/2.路网文件/xodr/UrbanRoad.xodr\" />\n\t\t<SceneGraphFile filepath=\"/home/intel/VIRES/VTD.2023.4/Data/Projects/CIDAS/scenario_library_Zhongqiyan/osgb/UrbanRoad.osgb\" />\n\t</RoadNetwork>\n\t<Entities>\n\t\t<ScenarioObject name=\"Ego\">\n\t\t\t<Vehicle name=\"Audi_A6_2010_black\" vehicleCategory=\"car\">\n\t\t\t\t<ParameterDeclarations />\n\t\t\t\t<Performance maxSpeed=\"69.444\" maxAcceleration=\"200\" maxDeceleration=\"10.0\" />\n\t\t\t\t<BoundingBox>\n\t\t\t\t\t<Center x=\"1.5\" y=\"0.0\" z=\"0.9\" />\n\t\t\t\t\t<Dimensions width=\"2.1\" length=\"4.5\" height=\"1.8\" />\n\t\t\t\t</BoundingBox>\n\t\t\t\t<Axles>\n\t\t\t\t\t<FrontAxle maxSteering=\"0.5\" wheelDiameter=\"0.6\" trackWidth=\"1.8\" positionX=\"3.1\" positionZ=\"0.3\" />\n\t\t\t\t\t<RearAxle maxSteering=\"0.0\" wheelDiameter=\"0.6\" trackWidth=\"1.8\" positionX=\"0.0\" positionZ=\"0.3\" />\n\t\t\t\t</Axles>\n\t\t\t\t<Properties>\n\t\t\t\t\t<Property name=\"control\" value=\"external\" />\n\t\t\t\t</Properties>\n\t\t\t</Vehicle>\n\t\t\t<ObjectController>\n\t\t\t\t<Controller name=\"DefaultDriver\">\n\t\t\t\t\t<ParameterDeclarations />\n\t\t\t\t\t<Properties />\n\t\t\t\t</Controller>\n\t\t\t</ObjectController>\n\t\t</ScenarioObject>\n\t\t<ScenarioObject name=\"Obj1\">\n\t\t\t<Vehicle name=\"ApriliaSR50_05_BlackGreen\" vehicleCategory=\"motorbike\">\n\t\t\t\t<ParameterDeclarations />\n\t\t\t\t<Performance maxSpeed=\"69.444\" maxAcceleration=\"200\" maxDeceleration=\"10.0\" />\n\t\t\t\t<BoundingBox>\n\t\t\t\t\t<Center x=\"1.5\" y=\"0.0\" z=\"0.9\" />\n\t\t\t\t\t<Dimensions width=\"2.1\" length=\"4.5\" height=\"1.8\" />\n\t\t\t\t</BoundingBox>\n\t\t\t\t<Axles>\n\t\t\t\t\t<FrontAxle maxSteering=\"0.5\" wheelDiameter=\"0.54\" trackWidth=\"0\" positionX=\"3.1\" positionZ=\"0.3\" />\n\t\t\t\t\t<RearAxle maxSteering=\"0.0\" wheelDiameter=\"0.54\" trackWidth=\"0\" positionX=\"0.0\" positionZ=\"0.3\" />\n\t\t\t\t</Axles>\n\t\t\t\t<Properties>\n\t\t\t\t\t<Property name=\"control\" value=\"internal\" />\n\t\t\t\t</Properties>\n\t\t\t</Vehicle>\n\t\t\t<ObjectController>\n\t\t\t\t<Controller name=\"DefaultDriver\">\n\t\t\t\t\t<ParameterDeclarations />\n\t\t\t\t\t<Properties />\n\t\t\t\t</Controller>\n\t\t\t</ObjectController>\n\t\t</ScenarioObject>\n\t</Entities>\n\t<Storyboard>\n\t\t<Init>\n\t\t\t<Actions>\n\t\t\t\t<Private entityRef=\"Ego\">\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"5.555556\" />\n\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<TeleportAction>\n\t\t\t\t\t\t\t<Position>\n\t\t\t\t\t\t\t\t<LanePosition roadId=\"0\" laneId=\"-2\" offset=\"0.000000\" s=\"100.000000\">\n\t\t\t\t\t\t\t\t\t<Orientation type=\"relative\" h=\"0.000000\" p=\"0.000000\" r=\"0.000000\" />\n\t\t\t\t\t\t\t\t</LanePosition>\n\t\t\t\t\t\t\t</Position>\n\t\t\t\t\t\t</TeleportAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t</Private>\n\t\t\t\t<Private entityRef=\"Obj1\">\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"0.000000\" />\n\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<TeleportAction>\n\t\t\t\t\t\t\t<Position>\n\t\t\t\t\t\t\t\t<LanePosition roadId=\"0\" laneId=\"-2\" offset=\"0.000000\" s=\"270.000000\">\n\t\t\t\t\t\t\t\t\t<Orientation type=\"relative\" h=\"0.000000\" p=\"0.000000\" r=\"0.000000\" />\n\t\t\t\t\t\t\t\t</LanePosition>\n\t\t\t\t\t\t\t</Position>\n\t\t\t\t\t\t</TeleportAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t</Private>\n\t\t\t</Actions>\n\t\t</Init>\n\t\t<Story name=\"\">\n\t\t\t<Act name=\"Behavior\">\n\t\t\t\t<ManeuverGroup maximumExecutionCount=\"1\" name=\"Ego_ManeuverGroup\">\n\t\t\t\t\t<Actors selectTriggeringEntities=\"false\">\n\t\t\t\t\t\t<EntityRef entityRef=\"Ego\" />\n\t\t\t\t\t</Actors>\n\t\t\t\t\t<Maneuver name=\"Ego_Maneuver\">\n\t\t\t\t\t\t<Event name=\"Ego_Maneuver_event_0\" priority=\"skip\" maximumExecutionCount=\"1\">\n\t\t\t\t\t\t\t<Action name=\"Ego_action_speed\">\n\t\t\t\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0.000000\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"5.555556\" />\n\t\t\t\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t\t\t</Action>\n\t\t\t\t\t\t\t<StartTrigger>\n\t\t\t\t\t\t\t\t<ConditionGroup>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Ego_Maneuver_event_0_event_codition_simulationtime\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t\t\t\t\t\t<SimulationTimeCondition value=\"0.000000\" rule=\"greaterThan\" />\n\t\t\t\t\t\t\t\t\t\t</ByValueCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t</ConditionGroup>\n\t\t\t\t\t\t\t</StartTrigger>\n\t\t\t\t\t\t</Event>\n\t\t\t\t\t</Maneuver>\n\t\t\t\t</ManeuverGroup>\n\t\t\t\t<ManeuverGroup maximumExecutionCount=\"1\" name=\"Obj1_ManeuverGroup\">\n\t\t\t\t\t<Actors selectTriggeringEntities=\"false\">\n\t\t\t\t\t\t<EntityRef entityRef=\"Obj1\" />\n\t\t\t\t\t</Actors>\n\t\t\t\t\t<Maneuver name=\"Obj1_Maneuver\">\n\t\t\t\t\t\t<Event name=\"Obj1_Maneuver_event_0\" priority=\"skip\" maximumExecutionCount=\"1\">\n\t\t\t\t\t\t\t<Action name=\"Obj1_action_speed\">\n\t\t\t\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0.000000\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"0.000000\" />\n\t\t\t\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t\t\t</Action>\n\t\t\t\t\t\t\t<StartTrigger>\n\t\t\t\t\t\t\t\t<ConditionGroup>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Obj1_Maneuver_event_0_event_codition_simulationtime\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t\t\t\t\t\t<SimulationTimeCondition value=\"0.000000\" rule=\"greaterThan\" />\n\t\t\t\t\t\t\t\t\t\t</ByValueCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t</ConditionGroup>\n\t\t\t\t\t\t\t</StartTrigger>\n\t\t\t\t\t\t</Event>\n\t\t\t\t\t\t<Event name=\"Obj1_Maneuver_event_1\" priority=\"skip\" maximumExecutionCount=\"1\">\n\t\t\t\t\t\t\t<Action name=\"Obj1_action_speed\">\n\t\t\t\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"linear\" value=\"1.680108\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"4.166667\" />\n\t\t\t\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t\t\t</Action>\n\t\t\t\t\t\t\t<StartTrigger>\n\t\t\t\t\t\t\t\t<ConditionGroup>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Obj1_Maneuver_event_1_event_condition_storyboard_elementstate\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t\t\t\t\t\t<StoryboardElementStateCondition storyboardElementType=\"event\" storyboardElementRef=\"Obj1_Maneuver_event_0\" state=\"completeState\" />\n\t\t\t\t\t\t\t\t\t\t</ByValueCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Obj1_Maneuver_event_1_event_condition_relativedistance\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByEntityCondition>\n\t\t\t\t\t\t\t\t\t\t\t<TriggeringEntities triggeringEntitiesRule=\"any\">\n\t\t\t\t\t\t\t\t\t\t\t\t<EntityRef entityRef=\"Ego\" />\n\t\t\t\t\t\t\t\t\t\t\t</TriggeringEntities>\n\t\t\t\t\t\t\t\t\t\t\t<EntityCondition>\n\t\t\t\t\t\t\t\t\t\t\t\t<RelativeDistanceCondition entityRef=\"Obj1\" relativeDistanceType=\"cartesianDistance\" freespace=\"false\" rule=\"lessThan\" value=\"19.059999\" />\n\t\t\t\t\t\t\t\t\t\t\t</EntityCondition>\n\t\t\t\t\t\t\t\t\t\t</ByEntityCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t</ConditionGroup>\n\t\t\t\t\t\t\t</StartTrigger>\n\t\t\t\t\t\t</Event>\n\t\t\t\t\t</Maneuver>\n\t\t\t\t</ManeuverGroup>\n\t\t\t</Act>\n\t\t</Story>\n\t\t<StopTrigger>\n\t\t\t<ConditionGroup>\n\t\t\t\t<Condition name=\"stop\" delay=\"0\" conditionEdge=\"rising\">\n\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t<SimulationTimeCondition value=\"70\" rule=\"greaterThan\" />\n\t\t\t\t\t</ByValueCondition>\n\t\t\t\t</Condition>\n\t\t\t</ConditionGroup>\n\t\t</StopTrigger>\n\t</Storyboard>\n</OpenSCENARIO>"

Task: Generate OpenScenario v2.0 osc file for the given task. 
Example 1
Input: Please generate / create an openscenario v2.0 osc file, requirements: An electric scooter moves in the same direction as the vehicle at a speed of 15 km/h, at a distance of 19m. The vehicle is traveling at 20 km/h to test for collision.
Output: "# \u6d4b\u8bd5\u573a\u666f: \u81ea\u884c\u8f66\u5de6\u4fa7\u907f\u8ba9 (CBLA - Crossing Bicycle Left Avoidance)\n# \u63cf\u8ff0: \u6d4b\u8bd5\u81ea\u884c\u8f66\u6cbf\u9053\u8def\u884c\u9a76\u65f6\u7684\u907f\u8ba9\u884c\u4e3a\n# \u573a\u666f: \u81ea\u8f66\u884c\u9a76\u4e2d,\u884c\u4eba\u4ee515kph\u6cbf\u9053\u8def\u6700\u53f3\u4fa7\u8f66\u9053\u79fb\u52a8\n\nimport \"$PROJECT_DIR/projectLibs/project_scenario_base/project_base_scenario_model_ssp.osc\"\nimport \"$OSC2LIB/scenarios/free_drive/ego_free_drive_auto_gen/ego_free_drive_auto_gen_one_way_road_top.osc\"\n\nextend test_config:\n    set map = \"$OSC2LIB/maps/M499_FTX_suburban.xodr\"\n\nextend vru_driver:\n    keep(default vru_behavior.enable == false)\n\nscenario sut.cbla_test_run inherits project_base_scenario:\n    keep(default scenario_str_generic == \"cbla_test_run\")\n\n    # \u6d4b\u8bd5\u9053\u8def\n    road: one_way_road\n    \n    # VRU\u884c\u4eba\n    vru: cyclist\n    \n    # \u8d77\u59cb\u548c\u7ed3\u675f\u4f4d\u7f6e\n    start_position: msp_position\n    end_position: msp_position\n\n    # \u8d77\u59cb\u504f\u79fb\n    start_offset: length with:\n        keep(it <= road.length - 100m)\n\n    # \u81ea\u8f66\u8d77\u59cb\u901f\u5ea6\n    start_speed: speed with:\n        keep(default it in [20kph, 30kph, 40kph, 50kph, 60kph])\n\n    # VRU\u901f\u5ea6\n    vru_speed: speed with:\n        keep(default it == 15kph)\n\n\n    # VRU\u8d77\u59cb\u504f\u79fb\n    vru_start_offset: length with:\n        keep(default it == 6m)\n\n    # \u6d4b\u8bd5\u6301\u7eed\u65f6\u95f4\n    duration_time: time with:\n        keep(it >= vru_start_offset / vru_speed + 2s)\n        keep(it <= 3 * vru_start_offset / vru_speed + 2s)\n\n    # VRU\u8d77\u59cb\u4f4d\u7f6e (\u6700\u53f3\u4fa7\u8f66\u9053)\n    position_along_road(\n        start_position, \n        road,\n        lon_offset: start_offset + vru_start_offset,\n        lat_offset: 0m, \n        rightmost_lane: true\n    )\n\n    # VRU\u7ed3\u675f\u4f4d\u7f6e\n    position_along_road(\n        end_position, \n        road,\n        lon_offset: start_offset + vru_start_offset + duration_time * vru_speed,\n        lat_offset: 0m, \n        rightmost_lane: true\n    )\n\n    keep(sut.car.ftx_driver.bypass_behavior.allow_opposite_left_side == true)\n\n    on @set_up.start:\n        call logger.log_info(\"start_speed       : $(start_speed)\")\n        call logger.log_info(\"vru_speed         : $(vru_speed)\")\n        call logger.log_info(\"vru_start_offset  : $(vru_start_offset)\")\n        call logger.log_info(\"duration_time     : $(duration_time)\")\n\n    do serial():\n        set_up: parallel():\n            # \u81ea\u8f66\u81ea\u7531\u884c\u9a76\n            ego_free_drive_phase: sut.ego_free_drive_auto_gen_one_way_road() with:\n                keep(it.gen_ego_speed_at_start == start_speed)\n                keep(it.ego_road_element == road)\n                keep(it.gen_min_lanes == 2)\n                keep(it.gen_max_lanes == 2)\n                keep(it.gen_ego_lane_at_start == 2)\n                keep(it.ego_start_offset == start_offset)\n                keep(it.ego_driving_duration == duration_time)\n\n            # VRU\u6cbf\u9053\u8def\u79fb\u52a8\n            vru.move(end_position, start_position, duration: duration_time)\n\nextend top.main:\n    do sut.cbla_test_run()\n\nextend test_additional_parameters:\n    const test_index: string\n\nextend test_config:\n    set test_name = \"cbla_test_run\"\n    set additional_parameters.test_index = \"auto_20251106_075241\"\n"

Task:'''

openscenario_gen_xml_template = '''Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: Generate OpenScenario v1.0 XML file for the given task. 
Example 1
Input: Please generate / create an openscenario v1.0 xml file, requirements: An electric scooter moves in the same direction as the vehicle at a speed of 15 km/h, at a distance of 19m. The vehicle is traveling at 20 km/h to test for collision.
Output: "<?xml version='1.0' encoding='utf-8'?>\n<OpenSCENARIO>\n\t<FileHeader revMajor=\"1\" revMinor=\"0\" date=\"2023-09-22T10:03:19\" description=\"CNCAP2021AEBVRUTWCBLA50_Bicycle_TargetSlow_impact50\" author=\"catarc\" />\n\t<ParameterDeclarations />\n\t<CatalogLocations>\n\t\t<VehicleCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/Vehicles\" />\n\t\t</VehicleCatalog>\n\t\t<PedestrianCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/Pedestrians\" />\n\t\t</PedestrianCatalog>\n\t\t<MiscObjectCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/Objects\" />\n\t\t</MiscObjectCatalog>\n\t\t<ControllerCatalog>\n\t\t\t<Directory path=\"Distros/Current/Config/Players/driverCfg.xml\" />\n\t\t</ControllerCatalog>\n\t\t<RouteCatalog>\n\t\t\t<Directory path=\"Data/Projects/Current/Scenarios/RouteCatalogs\" />\n\t\t</RouteCatalog>\n\t\t<TrajectoryCatalog>\n\t\t\t<Directory path=\"Data/Projects/Current/Scenarios/TrajectoryCatalogs\" />\n\t\t</TrajectoryCatalog>\n\t</CatalogLocations>\n\t<RoadNetwork>\n\t\t<LogicFile filepath=\"/Users/zhaozhijian/Dev/quanxi/esmini/bin/xml_test/2.路网文件/xodr/UrbanRoad.xodr\" />\n\t\t<SceneGraphFile filepath=\"/home/intel/VIRES/VTD.2023.4/Data/Projects/CIDAS/scenario_library_Zhongqiyan/osgb/UrbanRoad.osgb\" />\n\t</RoadNetwork>\n\t<Entities>\n\t\t<ScenarioObject name=\"Ego\">\n\t\t\t<Vehicle name=\"Audi_A6_2010_black\" vehicleCategory=\"car\">\n\t\t\t\t<ParameterDeclarations />\n\t\t\t\t<Performance maxSpeed=\"69.444\" maxAcceleration=\"200\" maxDeceleration=\"10.0\" />\n\t\t\t\t<BoundingBox>\n\t\t\t\t\t<Center x=\"1.5\" y=\"0.0\" z=\"0.9\" />\n\t\t\t\t\t<Dimensions width=\"2.1\" length=\"4.5\" height=\"1.8\" />\n\t\t\t\t</BoundingBox>\n\t\t\t\t<Axles>\n\t\t\t\t\t<FrontAxle maxSteering=\"0.5\" wheelDiameter=\"0.6\" trackWidth=\"1.8\" positionX=\"3.1\" positionZ=\"0.3\" />\n\t\t\t\t\t<RearAxle maxSteering=\"0.0\" wheelDiameter=\"0.6\" trackWidth=\"1.8\" positionX=\"0.0\" positionZ=\"0.3\" />\n\t\t\t\t</Axles>\n\t\t\t\t<Properties>\n\t\t\t\t\t<Property name=\"control\" value=\"external\" />\n\t\t\t\t</Properties>\n\t\t\t</Vehicle>\n\t\t\t<ObjectController>\n\t\t\t\t<Controller name=\"DefaultDriver\">\n\t\t\t\t\t<ParameterDeclarations />\n\t\t\t\t\t<Properties />\n\t\t\t\t</Controller>\n\t\t\t</ObjectController>\n\t\t</ScenarioObject>\n\t\t<ScenarioObject name=\"Obj1\">\n\t\t\t<Vehicle name=\"ApriliaSR50_05_BlackGreen\" vehicleCategory=\"motorbike\">\n\t\t\t\t<ParameterDeclarations />\n\t\t\t\t<Performance maxSpeed=\"69.444\" maxAcceleration=\"200\" maxDeceleration=\"10.0\" />\n\t\t\t\t<BoundingBox>\n\t\t\t\t\t<Center x=\"1.5\" y=\"0.0\" z=\"0.9\" />\n\t\t\t\t\t<Dimensions width=\"2.1\" length=\"4.5\" height=\"1.8\" />\n\t\t\t\t</BoundingBox>\n\t\t\t\t<Axles>\n\t\t\t\t\t<FrontAxle maxSteering=\"0.5\" wheelDiameter=\"0.54\" trackWidth=\"0\" positionX=\"3.1\" positionZ=\"0.3\" />\n\t\t\t\t\t<RearAxle maxSteering=\"0.0\" wheelDiameter=\"0.54\" trackWidth=\"0\" positionX=\"0.0\" positionZ=\"0.3\" />\n\t\t\t\t</Axles>\n\t\t\t\t<Properties>\n\t\t\t\t\t<Property name=\"control\" value=\"internal\" />\n\t\t\t\t</Properties>\n\t\t\t</Vehicle>\n\t\t\t<ObjectController>\n\t\t\t\t<Controller name=\"DefaultDriver\">\n\t\t\t\t\t<ParameterDeclarations />\n\t\t\t\t\t<Properties />\n\t\t\t\t</Controller>\n\t\t\t</ObjectController>\n\t\t</ScenarioObject>\n\t</Entities>\n\t<Storyboard>\n\t\t<Init>\n\t\t\t<Actions>\n\t\t\t\t<Private entityRef=\"Ego\">\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"5.555556\" />\n\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<TeleportAction>\n\t\t\t\t\t\t\t<Position>\n\t\t\t\t\t\t\t\t<LanePosition roadId=\"0\" laneId=\"-2\" offset=\"0.000000\" s=\"100.000000\">\n\t\t\t\t\t\t\t\t\t<Orientation type=\"relative\" h=\"0.000000\" p=\"0.000000\" r=\"0.000000\" />\n\t\t\t\t\t\t\t\t</LanePosition>\n\t\t\t\t\t\t\t</Position>\n\t\t\t\t\t\t</TeleportAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t</Private>\n\t\t\t\t<Private entityRef=\"Obj1\">\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"0.000000\" />\n\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t<TeleportAction>\n\t\t\t\t\t\t\t<Position>\n\t\t\t\t\t\t\t\t<LanePosition roadId=\"0\" laneId=\"-2\" offset=\"0.000000\" s=\"270.000000\">\n\t\t\t\t\t\t\t\t\t<Orientation type=\"relative\" h=\"0.000000\" p=\"0.000000\" r=\"0.000000\" />\n\t\t\t\t\t\t\t\t</LanePosition>\n\t\t\t\t\t\t\t</Position>\n\t\t\t\t\t\t</TeleportAction>\n\t\t\t\t\t</PrivateAction>\n\t\t\t\t</Private>\n\t\t\t</Actions>\n\t\t</Init>\n\t\t<Story name=\"\">\n\t\t\t<Act name=\"Behavior\">\n\t\t\t\t<ManeuverGroup maximumExecutionCount=\"1\" name=\"Ego_ManeuverGroup\">\n\t\t\t\t\t<Actors selectTriggeringEntities=\"false\">\n\t\t\t\t\t\t<EntityRef entityRef=\"Ego\" />\n\t\t\t\t\t</Actors>\n\t\t\t\t\t<Maneuver name=\"Ego_Maneuver\">\n\t\t\t\t\t\t<Event name=\"Ego_Maneuver_event_0\" priority=\"skip\" maximumExecutionCount=\"1\">\n\t\t\t\t\t\t\t<Action name=\"Ego_action_speed\">\n\t\t\t\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0.000000\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"5.555556\" />\n\t\t\t\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t\t\t</Action>\n\t\t\t\t\t\t\t<StartTrigger>\n\t\t\t\t\t\t\t\t<ConditionGroup>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Ego_Maneuver_event_0_event_codition_simulationtime\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t\t\t\t\t\t<SimulationTimeCondition value=\"0.000000\" rule=\"greaterThan\" />\n\t\t\t\t\t\t\t\t\t\t</ByValueCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t</ConditionGroup>\n\t\t\t\t\t\t\t</StartTrigger>\n\t\t\t\t\t\t</Event>\n\t\t\t\t\t</Maneuver>\n\t\t\t\t</ManeuverGroup>\n\t\t\t\t<ManeuverGroup maximumExecutionCount=\"1\" name=\"Obj1_ManeuverGroup\">\n\t\t\t\t\t<Actors selectTriggeringEntities=\"false\">\n\t\t\t\t\t\t<EntityRef entityRef=\"Obj1\" />\n\t\t\t\t\t</Actors>\n\t\t\t\t\t<Maneuver name=\"Obj1_Maneuver\">\n\t\t\t\t\t\t<Event name=\"Obj1_Maneuver_event_0\" priority=\"skip\" maximumExecutionCount=\"1\">\n\t\t\t\t\t\t\t<Action name=\"Obj1_action_speed\">\n\t\t\t\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"step\" value=\"0.000000\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"0.000000\" />\n\t\t\t\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t\t\t</Action>\n\t\t\t\t\t\t\t<StartTrigger>\n\t\t\t\t\t\t\t\t<ConditionGroup>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Obj1_Maneuver_event_0_event_codition_simulationtime\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t\t\t\t\t\t<SimulationTimeCondition value=\"0.000000\" rule=\"greaterThan\" />\n\t\t\t\t\t\t\t\t\t\t</ByValueCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t</ConditionGroup>\n\t\t\t\t\t\t\t</StartTrigger>\n\t\t\t\t\t\t</Event>\n\t\t\t\t\t\t<Event name=\"Obj1_Maneuver_event_1\" priority=\"skip\" maximumExecutionCount=\"1\">\n\t\t\t\t\t\t\t<Action name=\"Obj1_action_speed\">\n\t\t\t\t\t\t\t\t<PrivateAction>\n\t\t\t\t\t\t\t\t\t<LongitudinalAction>\n\t\t\t\t\t\t\t\t\t\t<SpeedAction>\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionDynamics dynamicsShape=\"linear\" value=\"1.680108\" dynamicsDimension=\"time\" />\n\t\t\t\t\t\t\t\t\t\t\t<SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t\t\t<AbsoluteTargetSpeed value=\"4.166667\" />\n\t\t\t\t\t\t\t\t\t\t\t</SpeedActionTarget>\n\t\t\t\t\t\t\t\t\t\t</SpeedAction>\n\t\t\t\t\t\t\t\t\t</LongitudinalAction>\n\t\t\t\t\t\t\t\t</PrivateAction>\n\t\t\t\t\t\t\t</Action>\n\t\t\t\t\t\t\t<StartTrigger>\n\t\t\t\t\t\t\t\t<ConditionGroup>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Obj1_Maneuver_event_1_event_condition_storyboard_elementstate\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t\t\t\t\t\t<StoryboardElementStateCondition storyboardElementType=\"event\" storyboardElementRef=\"Obj1_Maneuver_event_0\" state=\"completeState\" />\n\t\t\t\t\t\t\t\t\t\t</ByValueCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t\t<Condition name=\"Obj1_Maneuver_event_1_event_condition_relativedistance\" delay=\"0\" conditionEdge=\"none\">\n\t\t\t\t\t\t\t\t\t\t<ByEntityCondition>\n\t\t\t\t\t\t\t\t\t\t\t<TriggeringEntities triggeringEntitiesRule=\"any\">\n\t\t\t\t\t\t\t\t\t\t\t\t<EntityRef entityRef=\"Ego\" />\n\t\t\t\t\t\t\t\t\t\t\t</TriggeringEntities>\n\t\t\t\t\t\t\t\t\t\t\t<EntityCondition>\n\t\t\t\t\t\t\t\t\t\t\t\t<RelativeDistanceCondition entityRef=\"Obj1\" relativeDistanceType=\"cartesianDistance\" freespace=\"false\" rule=\"lessThan\" value=\"19.059999\" />\n\t\t\t\t\t\t\t\t\t\t\t</EntityCondition>\n\t\t\t\t\t\t\t\t\t\t</ByEntityCondition>\n\t\t\t\t\t\t\t\t\t</Condition>\n\t\t\t\t\t\t\t\t</ConditionGroup>\n\t\t\t\t\t\t\t</StartTrigger>\n\t\t\t\t\t\t</Event>\n\t\t\t\t\t</Maneuver>\n\t\t\t\t</ManeuverGroup>\n\t\t\t</Act>\n\t\t</Story>\n\t\t<StopTrigger>\n\t\t\t<ConditionGroup>\n\t\t\t\t<Condition name=\"stop\" delay=\"0\" conditionEdge=\"rising\">\n\t\t\t\t\t<ByValueCondition>\n\t\t\t\t\t\t<SimulationTimeCondition value=\"70\" rule=\"greaterThan\" />\n\t\t\t\t\t</ByValueCondition>\n\t\t\t\t</Condition>\n\t\t\t</ConditionGroup>\n\t\t</StopTrigger>\n\t</Storyboard>\n</OpenSCENARIO>"

Task:'''

openscenario_agent_function_call_template = '''
Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

**Task:** Generate agent function call task data.

**Constraints:**
1. The output must be formatted as a JSON object with the following keys:
   - `"name"`: the function name (must be one of: `read_file`, `edit_file`, `delete_file`, `create_file`, `list_directory`, `grep_search`, `file_search`, `run_terminal_command`)
   - `"arguments"`: a dictionary of function arguments, where each key is the argument name and the value is the corresponding value.

2. Function descriptions and their required/optional arguments:
   - `read_file`: Read the contents of a file.
     - `target_file` (required): The path of the file to read.
     - `offset` (optional): The line number to start reading from (1-indexed).
     - `limit` (optional): The number of lines to read.
     - `should_read_entire_file` (optional): Whether to read the entire file.
     - `agent` (optional): Reference to the agent instance for permission checks.
   - `edit_file`: Edit a file according to the provided instructions and code changes.
     - `target_file` (required): Path to the file to edit.
     - `instructions` (required): Instructions describing the edit.
     - `code_edit` (optional): Line-based edit as a JSON dictionary or string with line ranges as keys: `{"1-5": "new content", "10-12": "more content"}`.
     - `code_replace` (optional): Complete replacement content for the file (if `code_edit` not provided).
     - `agent` (optional): Reference to the agent instance for permission checks.
   - `delete_file`: Delete a file at the specified path.
     - `target_file` (required): The path of the file to delete.
     - `agent` (optional): Reference to the agent instance for permission checks.
   - `create_file`: Create a new file with the given content.
     - `file_path` (required): Path where the file should be created.
     - `content` (required): Content to write to the file.
     - `agent` (optional): Reference to the agent instance for permission checks.
   - `list_directory`: List the contents of a directory.
     - `relative_workspace_path` (required): Path to list contents of.
     - `agent` (optional): Reference to the agent instance for permission checks.
   - `grep_search`: Fast text-based regex search that finds exact pattern matches within files or directories.
     - `query` (required): The regex pattern to search for.
     - `explanation` (optional): Optional explanation of why this search is being performed.
     - `case_sensitive` (optional): Whether the search should be case sensitive.
     - `include_pattern` (optional): Optional glob pattern for files to include.
     - `exclude_pattern` (optional): Optional glob pattern for files to exclude.
     - `agent` (optional): Reference to the agent instance (unused in this function but kept for consistency).
   - `file_search`: Fast file search based on fuzzy matching against file path.
     - `query` (required): Fuzzy filename to search for.
     - `explanation` (optional): Optional explanation of why this search is being performed.
     - `agent` (optional): Reference to the agent instance (unused in this function but kept for consistency).
   - `run_terminal_command`: Run a terminal command.
     - `command` (required): The terminal command to execute
     - `explanation` (optional): Optional explanation of why this command needs to be run
     - `is_background` (optional): Whether the command should be run in the background
     - `require_user_approval` (optional): Whether the user must approve the command before execution
     - `agent` (optional): Reference to the agent instance for permission checks
3. The output must be formatted data, including the key name for the function name and the key arguments for the corresponding function parameters, as shown below:
{
  "name": "xxx",
  "arguments": {
    "xxx": "xxx",
    ...
  }
}

**Example 1**  
Input: Let me check what other files might be relevant for lane changing scenarios:
Output:  {\"name\": \"grep_search\", \"arguments\": {\"query\": \"change.*lane|lane.*change\", \"include_pattern\": \"rag_osc/standard_test_scenarios/*.osc\"}}

**Example 2**  
Input: Now I understand the structure. Based on the requirements and examples, I'll create a DSL code file for the specified test scenario. Let me create the CDA_002.osc file with the proper structure:
Output:  {\"name\": \"create_file\", \"arguments\": {\"file_path\": \"CDA_002.osc\", \"content\": \"# Test scenario: Highway curve with lane change\\n# Description: Test vehicle stability on curves and lane changing before curve entry\\n# Scenario: Highway curve with vehicle changing lanes before entering curve\\n\\nimport \\\"$PROJECT_DIR/projectLibs/project_scenario_base/project_base_scenario_model_ssp.osc\\\"\\nimport \\\"$OSC2LIB/scenarios/free_drive/curved_road/ego_free_drive_on_curved_road.osc\\\"\\nimport \\\"$OSC2LIB/scenarios/free_drive/highway/ego_free_drive_on_highway.osc\\\"\\nimport \\\"$OSC2LIB/scenarios/free_drive/npc_driving/npc_free_driving_top.osc\\\"\\n\\nextend test_config:\\n    set map = \\\"$OSC2LIB/maps/M499_FTX_suburban.xodr\\\"\\n\\nscenario sut.highway_curve_lane_change_test inherits project_base_scenario:\\n    keep(default scenario_str_generic == \\\"highway_curve_lane_change_test\\\")\\n    \\n    in_road: highway\\n    \\n    # Ego vehicle speed\\n    ego_speed: speed with:\\n        keep(default it == 60kph)\\n    \\n    # Curve parameters\\n    curve_direction: string\\n    keep(default curve_direction == \\\"right_curve\\\")\\n    \\n    curve_radius: length with:\\n        keep(default it == 100m)\\n        \\n    curve_length: length with:\\n        keep(default it == 150m)\\n    \\n    # Lane change parameters\\n    initial_lane: int with:\\n        keep(default it == -2)  # Leftmost lane (outside lane)\\n        \\n    target_lane: int with:\\n        keep(default it == -1)  # Next lane inside\\n        \\n    # Trigger condition\\n    trigger_distance: length with:\\n        keep(default it == 80.0m)\\n    \\n    # Test duration\\n    duration_time: time with:\\n        keep(default it == 20s)\\n        \\n    # NPC vehicle type\\n    npc_vehicle_type: vehicle_category with:\\n        keep(default it == sedan)\\n    \\n    do parallel():\\n        set_up: serial():\\n            sut.car.drive() with:\\n                along(in_road)\\n                \\n        parallel():\\n            # Ego vehicle driving on highway before curve\\n            ego_phase: sut.ego_free_drive_on_highway() with:\\n                keep(it.gen_ego_lane_at_start == initial_lane)\\n                keep(it.gen_ego_lane_at_end == target_lane)\\n                keep(it.gen_ego_speed_at_start == ego_speed)\\n                keep(it.gen_ego_speed_at_end == ego_speed)\\n                keep(it.ego_driving_duration == duration_time)\\n                keep(it.ego_time_gap_to_start_of_road_element == 0.5s)\\n                \\n            # Curve section\\n            curve_phase: sut.ego_free_drive_on_curved_road() with:\\n                keep(it.gen_curve_direction == curve_direction)\\n                keep(it.gen_max_radius == curve_radius)\\n                keep(it.gen_min_length == curve_length)\\n                keep(it.gen_start_offset == trigger_distance)\\n                \\n            # NPC vehicle in curve\\n            npc_phase: sut.npc_free_driving() with:\\n                keep(it.npc_vehicle_type == self.npc_vehicle_type)\\n                keep(it.npc_relative_position == 30m)\\n                keep(it.npc_initial_speed == 0kph)\\n                keep(it.npc_target_speed == 0kph)\\n                keep(it.npc_duration == duration_time)\\n                keep(it.ref_car == sut.car)\\n\\nextend top.main:\\n    do sut.highway_curve_lane_change_test()\\n\\nextend test_additional_parameters:\\n    const test_index: string\\n\\nextend test_config:\\n    set test_name = \\\"highway_curve_lane_change_test\\\"\\n    set additional_parameters.test_index = \\\"auto_20251126_000000\\\"\"}

Task:'''
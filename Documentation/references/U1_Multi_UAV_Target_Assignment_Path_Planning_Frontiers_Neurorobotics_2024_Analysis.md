# U1shouldusesscoreanalysis: multiplenopersonmachineobjectivescoreallocationandpathpathplanning

**Full Citation**: "Multi-UAV simultaneous target assignment and path planning based on deep reinforcement learning in dynamic multiple-obstacles environments"

---

## 📄 Application Basic Information

* **Application Domain**: allocationsend/searchrescue/patrolinspectthroughuses"multiplemachineobjectivescoreallocation+pathpathplanning" (frameworkunitsthroughuses, experimentswithtoreachobjectivemostshortenpathismain). seecitelanguageandtaskfixedmeaning. 
* **System Scale**: trainingandtesttrialconstantuses **5 UAV × 5 objective**; statisticsexperimentsextensionto **3–7 units UAV** and **10–30 individualobstacleobject** (Fig15, p.17; Table2–3, p.12). 
* **Optimization Objective**: **minimizetotalflightprocess**, samewhensatisfy**completeallobjectivescoreallocation**and**collision avoidanceconstraint** (equation(1)–(3), p.3; frameworkunitsFig Fig.5, p.6). 
* **Algorithm Type**: **Actor-Critic (TD3)+ objectivescoreallocationnetwork**; scoreallocationstandardsignwith **Q valuematrix + Hungarybenefitmethod**getto (§4.1–4.2; Fig.5–6, p.6–9). 

---

## 🚁 UAVsystemmodelingscoreanalysis

1. **Airspace Modeling**

* **spacestructure**: **3D continuousspace**, taskdomainis **2×2×2 establishmethodbody** (Fig.7, p.7; §5.1, p.10). 
* **Altitude Processing**: **continuoushighdegree** (actionisthreeaxisforce/addspeeddegreecontrol, nondiscretelayer)(Fig.3B, p.5). 
* **Conflict Avoidance**: through**geometricdistancedistanceconstraint**andrewardpenaltyimplementationformachine-machine, machine-barriercollision avoidance (equation(3) andreward(6)(7), p.3, p.5–6). 

2. **Task Scheduling Mode**

* **scoreallocationstrategy**: **setinequationstandardsignalivebecome + removeincenterizationexecuterow**. eachstepsisplacehave UAV computetoeachobjective **Q valuematrix**, uses**Hungarybenefitmethod**gettooneonematchallocationasscoreallocationstandardsign, samewhentraining"objectivescoreallocationnetwork"outputgeneralrate (Fig.6 andequation(17)–(19), p.8–9). 
* **movestateweightscheduling**: **completeallmovestate** (graduallystepsweightscoreallocation, Fig.6 toppartflowprocess, p.8–9). 
* **load balancing**: notshowequationdesignmeanbalanceMetrics, mainlywith**allcovercoverscoreallocation** (complete assignment)andcollision avoidanceismain (p.3, p.8). 

3. **systemconstraint**

* **capacitylimitation**: loadweight/canquantitynotmodeling; **feelknowhalfpath** d_det=0.5 (Fig.7 and §5.1, p.7, p.10). 
* **whenbetweenconstraint**: eachreturncombine 100 step (Table1 exceedparameter, p.10). 
* **spaceconstraint**: obstacleobjectstatic/move (halfpath 0.1), edgeboundaryreverseelastic; collision avoidancehalfpathconstraint (equation(3), p.3; §5.1, p.10). 

---

## 🔍 andour"verticalscorelayerqueueizationsystem"Comparison

### discussionpapermethodneedpoint (provideComparison)

* **POMDP modeling**: localobservation + partscorecansee (Fig.2, §2.2, p.4). 
* **statespace**: internalstate + andobjective/hemachine/obstaclephaseforquantity; **dimensionaldegreepublicequation**is *(7 + 4(NU−1) + 4NT + 7NO)* (Fig.6 inpartnetworkinput, p.9). 
* **action**: **continuouscontrol (3axisforce)** (Fig.3B, p.5). 
* **scoreallocationandplanning**: **TANet-TD3** samestepscompletescoreallocationandpathpath; Hungarybenefitmethodmaintainproof"oneforone" (Fig.5–6, p.6–9). 
* **performance**: inmovestate/hybridloopenvironmentmeansuperiorinforaccording, **completerateandreceiveconvergespeeddegreechangespeed up** (Fig.9; Table2–3, p.11–12). 

### ouruniquedesign (returncustomer)

* **5layerhighdegree {100,80,60,40,20m}**, **inverted pyramidcapacity {8,6,4,3,2}**
* **pressuretriggerlayerbetweenundersink/transfer** (congestiondrivemove)
* **29dimensionalstructureizationstate** (queue/toreach/service/scoreflow/loadetc.)
* **MCRPS/D/K queuenetwork** (relatedtoreach, randombatchquantity, Poissonscoreflow, statedependency, movestatetransfer, finitecapacity)

### systeminnovationpropertyComparison (1–10 score)

1. **whetherhaveverticalscorelayerUAVscheduling？** **2/10**

 * paperinhighdegree**continuous**, notdo**discretescorelayer/layercapacity**design (Fig.7, p.7). 
2. **whetherhaveinverted pyramidresourceallocationplacement？** **0/10**

 * nolayerlevelcapacityordifferenthighdegreethroughchannelresourcescoreallocationconcept. 
3. **whetherhavequeuetheorymodelingUAVsystem？** **1/10**

 * objectiveisgeometricpaththroughmostshortenandcollision avoidance; **notintroducingqueueing/congestionprocess**stochastic processmodeling. 
4. **whetherhavepressuretriggerlayerbetweentransfer？** **0/10**

 * nolayerbetweenmigrationshiftmechanism. 
5. **whetherhave≥29dimensionalstatespacedesign？** **6/10**

 * **havehighdimensionalstate** (takedecideinobjective/obstaclenumber), but**nonqueueization/scorelayerstructureization**; our 29 dimensionalchangesideweightsystemlearningMetrics (Fig.6 inputpublicequation, p.9). 

**existingworkworkclosefocus**: 3D operatemovelearningavoidbarrierand**objectivescoreallocation+pathpathplanning**onebodyization (Fig.5–6); movestateobstacle, partscorecansee, endtoend DRL. 
**ourinnovationpoint**: **verticalairspacequeueizationmanagement**, **scorelayercapacityoptimization**, **pressuretriggertransfer**, **queuenetwork(MCRPS/D/K)** and **highdimensionalstructureizationstate**. 

---

## 💡 forourstudyresearchvaluevalue

1. **shouldusesverificationvaluevalue**: discussionpaper 3D continuousairspaceand**movestateweightscoreallocation** (graduallystepsHungarybenefit, Fig.6)explain**movestatecongestion/conflict**frequencysend, **supportverticalscorelayerandcapacitycontrolmustneedproperty** (collision avoidanceonlyrelygeometric/localfeelknowindensesetscenarioretreatization, Fig.15 obstacleincreasemultiplewhencompleterateunderfall, p.17). 
2. **methodComparisonvaluevalue**: ourcanTreats **"scoreallocationnetwork+planning"** idea, replaceis **"scorelayerqueuestate+strategyhead"**, compare"geometric-equationscoreallocation"and"queue-equationscheduling"**efficiency/fairness/tailpartwhendelay**. 
3. **scenarioextensionvaluevalue**: treatits**movestateobstacle**extensionis**layercongestion/capacitylimitation**; treat**Hungarybenefit**replace/parallelis**scorelayercapacitymatchallocation** (e.g.eachlayermostlarge andsend K_l). 
4. **performancebaselinevaluevalue**: Treats **TANet-TD3** as"**noscorelayer/noqueue**"strong baseline, foraccordingour **MCRPS/D/K + DRL** **tailpartrisk(p95/p99whendelay)**, **overflow rate**and**crosslayermigrationshifttimesnumber**. 

---

## resultdiscussionandsuggestion

* **shouldusesinnovationdegree (phaseforexistingUAVstudyresearch)**: **6/10**. discussionpaperin"**samestepsscoreallocation+pathpath**, partscorecansee, movestateobstacle, multiplemachinecooperative"ondoedtieactualworkprocessandexperimentsverification (Fig.9, Table2–3, Fig.15), but**nottouchandverticalscorelayer/capacity/queueprocess**. 
* **oursuperiorpotentialcertainrecognize**: **significantlyimprovement** (inairspacegrouporganizeandtheorymodelingdimensionaldegreeobviousleadfirst). 

### toolbodycanimplementplaceforreceivepoint

* **treatitsmovestateweightscoreallocationmechanism→layerbetweenmovestatetransfer**: Treats"eachstepsscoreallocation"changeis"**layerinnerqueueing + exceedthresholdpressurestrongtriggercrosslayermigrationshift**", andretain TD3 continuouscontrolheaddosamelayermicroadjust. 
* **usesitsstateconstructexperience**: retainits"**phaseforquantityobservation**"as**bottomlayerfeelknow**, againstackaddour **29 dimensionalqueue/congestionfeature**; Comparisononlygeometricfeature vs queue+geometric. 
* **Metrics**: divide"completerate/flightprocess", newincrease**average/scorepositionetc.waitingwhendelay, layerinnerbenefitusesrate, crosslayerswitchchangetimesnumber, overflow rate**, changecanbodyappear MCRPS/D/K valuevalue. 

---

### proofdataspeedview (pagecode/FigTable)

* objective: minimizetotalflightprocess+completeallscoreallocation+collision avoidance (equation(1)–(3), p.3; Fig.5, p.6). 
* POMDP andlocalobservation (Fig.2, §2.2, p.4). 
* state/action/rewarddesign (Fig.3, p.5; equation(5)–(8), p.5–6). 
* TANet-TD3 frameworkunitsandHungarybenefitscoreallocation (Fig.5–6, equation(17)–(19), p.6–9). 
* loopenvironmentandexceedparameter (Fig.7, p.7; Table1, p.10). 
* training/testtrialandComparison (Fig.9, Table2–3, p.11–12; Fig.10–14, p.12–16). 
* extensionstatistics (Fig.15, p.17). 

e.g.fruityouhopelook, Icanwithtreat **"layercapacityconstraint + pressuretriggertransfer + TANet-TD3 continuouscontrolhead"** **simulationexperimentsclearsingleandMetricsTable**directaccordingyous 5 layerairspaceand 29 dimensionalstateexitoneversionComparisonmethodplan. 

---

**theoryinnovationrelateddegree**: **low** (mainlyisworkprocessshoulduses, lackfewsystemtheorymodeling)
**ourinnovationuniquepropertycertainrecognize**: **completeallunique** (inverticalscorelayerqueueizationmethodaspect)
**suggestionadjuststudyprioritizedlevel**: **important** (asUAVshouldusesbaselineimportantreference)

---

**Analysis Completion Date**: 2025-01-28 
**Analysis Quality**: Detailed analysis withobjectivescoreallocationandpathpathplanningonebodyizationmechanism 
**Recommended Use**: asmultipleUAVcooperativecontrolshouldusesbaseline, referenceTANet-TD3frameworkunitsandmovestateweightscoreallocationmechanism
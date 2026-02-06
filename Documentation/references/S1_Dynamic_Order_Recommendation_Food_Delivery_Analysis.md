# S1Literature Analysis: accordingrequiresfoodproductallocationsendmovestateordersinglerecommend

**Full Citation**: X. Wang, L. Wang, C. Dong, H. Ren, and K. Xing, "Reinforcement Learning-Based Dynamic Order Recommendation for On-Demand Food Delivery," Tsinghua Science and Technology, vol. 29, no. 2, pp. 356-367, 2024, DOI: 10.26599/TST.2023.9010041.

---

## 📄 Application Basic Information

* **Application Domain**: **allocationsend (OFD)**. studyresearch"movestateordersinglerecommend/indicatedispatch", averageplatformtowardrideraccordingorderrecommendordersingle, ridergrabsingle. seecitelanguageandFig1. 
* **System Scale**: **inscale (10–50)** (eachindividualdecisionwhenmomentmostmultipleconsider 20 nameriderparameterandsorting/recommend; trueactualnumberdataquantity 366 ten thousandexchangemutualsamplebook). see§5.1 exceedparameter `m_t=20` andnumberdataexplain. 
* **Optimization Objective**: **multi-objectiveweighted** (Eq.(1)): maximizealreadypassivebecomepower"grasptake"ordersinglenumber `num_t`, minimize"grabsingleconflictnumber" `μ_t`, minimize"pathpathlengthanddelayerrorincreasequantity" `Δl_t`, andinnosinglegrabtakewhengivefixedpenalty `g`. 

## 🚁 UAVsystemmodelingscoreanalysis (accordingdiscussionpaperinnercontainmapping)

1. **Airspace Modeling**

* **spacestructure**: **2D plane + movestateloopenvironment** (ordersingleholdcontinuetoreach/disappearlose, riderpositionholdcontinuechangeization). see§1–§3. 
* **Altitude Processing**: **fixedfixed/notmodeling** (noairspacelayertimes). 
* **Conflict Avoidance**: **rules/geometricconstraint**bodyappearintwolayer: ① ordersinglelayerconflict: sameoneordersinglepassivemultiplepersonsamewhengrab ("order-grabbing conflict"); ② pathpathlayerconstraint: followguardfirstback, capacity, whenbetweenwindowetc.pathrule (attachinpaththroughplanningconstraintdescribedescription). see§3, Table1andequation(1)–(3). 

2. **Task Scheduling Mode**

* **scoreallocationstrategy**: **setinequation**averageplatformintelligentbody (MDP)for**multiplerider**forwardorderplacealivebecomerecommendcolumnTable; adopting**LSTM codecode–solutioncodeequation actor-critic**, treatfirstfirstisOtherrideralivebecomeclearsingleembeddingtowhenfirstdecision, showequationsuppress"conflict". seeFig3–4and§4.1. 
* **movestateweightscheduling**: **completeallmovestate** (eachwhenmomentweightcalculatecolumnTable; training/onlineflowprocessseealgorithm1–2). 
* **load balancing**: noshowequationmeanbalanceitem, butthroughrewardin"conflictpenalty+pathpathincreasequantity"betweenreceiveimplementationsystemlayeraspectaveragebalance. 

3. **systemconstraint**

* **capacitylimitation**: ordersingle/riderloadload, whenbetweenwindow, firstbackconstraintetc. ("route planning constraints"). 
* **whenbetweenconstraint**: whenbetweenwindow/promisewhenlimit; traininginsettingeachreturncombinestepsgrow `T=40`. 
* **spaceconstraint**: based ontakesendpointthroughlatitudedegreedistancedistance/pathlinesubstitutevalue; stateinincludecontainindividualbodyandexchangemutualfeature (Table2). 

> keymechanismandEvidence: MDPmodelingandreward (§3, Eq.(1)), **LSTM actor-critic** (Fig3–4), **riderwhenorderrules** (accordingalreadygrabsinglenumber/becomepowerrate/randomthreetypesorting; §4.2), **experimentsMetrics** (TIS/CR/TNGO/TNOGC/TIRLTD)andComparisonresults (Table3), where"accordingbecomepowerratesorting"isbest (Fig5). 

## 🔍 andour"verticalscorelayerqueueizationUAVsystem"Comparison

### ouruniquedesign (returncustomer)

* 5 layerhighdegree {100,80,60,40,20m}; **inverted pyramidcapacity** {8,6,4,3,2}
* **congestionpressuretrigger**layerbetweenundersink/transfer
* **29dimensionalstate** (teamgrow/toreach/service/scoreflow/load…)
* **MCRPS/D/K** queuenetwork (multiplelayerrelatedtoreach, randombatchquantityservice, Poissonscoreflow, statedependency, movestatetransfer, finitecapacity)

### systeminnovationpropertyComparison (1–10 score)

1. **verticalscorelayerUAVscheduling？**: **0/10** (discussionpaperisplaceaspect 2D, notmodelinghighdegreelayer). 
2. **inverted pyramidresourceallocationplacement？**: **0/10** (nolayercapacity/throughchannelnumberconcept). 
3. **queuetheorymodelingUAVsystem？**: **1/10** (uses **MDP/RL**, notshowequationbuildestablishqueueing/congestionstochastic process). 
4. **pressuretriggerlayerbetweentransfer？**: **0/10** (nolayerbetweenmechanism; onlyhave"conflict"suppress). 
5. **≥29dimensionalstatespacedesign？**: **6/10** (Table2columnexitindividualbody+exchangemutual+ordersinglebelongpropertyetc.multiplefeature, dimensionaldegreecanhighin 29, but**nonscorelayer/queueization**structure). 

### shouldusesscenariopoordifference

**existingworkworkclosefocus**: **levelcooperative** (multiplerider), **pathpathsubstitutevalue/delayerror**, **conflictsuppress**, **movestaterecommend**and**multi-objectiveweighted**evaluates (Table3, Fig5). 
**ourinnovationpoint**: 

* ✅ **verticalairspacequeueizationmanagement** (layerandthroughchannelcapacityK)
* ✅ **scorelayercapacitymovestateoptimization** (inverted pyramid+canweightallocationplacement)
* ✅ **queueingdiscussiondrivemovedesign** (MCRPS/D/K)
* ✅ **highdimensionalsystemstate**and**pressuretriggercrosslayer**

## 💡 forourstudyresearchvaluevalue

1. **shouldusesverificationvaluevalue**: discussionpaperproofclear**movestate, multiplepersonphasemutualimpact**undersetinequationstrategyrequiresneed**forwardorderizationandmemory (LSTM)**comesuppressconflictandproposerisethroughput (Fig3–4, Fig5, Table3), thisandmultipleUAVindensesettask/hotpointarea"** andsendconflict**"highdegreephasesimilar, isourintroducing**forwardorderizationscoreallocation/scorelayerinnermemory**providedependdata. 

2. **methodComparisonvaluevalue**: canTreatsits **RLORM (LSTM actor-critic + ridersorting)** migrationshiftis"**layerinnertaskrecommenddevice**", andour"**scorelayercapacity+pressuretransfer**"stackadd/foraccording, quantityizationin**conflictrate, tailpartwhendelay (p95/p99), overflow rate**onimprovement (Metricsbodysystemcanrepeatuses TIS/CR/TNOGC/TIRLTD). 

3. **scenarioextensionvaluevalue**: treat"riderbecomepowerratesorting (Seq-succeed)"changecreateis"**layer/nopersonmachineprioritizedlevelsorting**" (e.g.accordinglayerinneretc.waitingwhengrow/loadGini/canusescanquantitysorting), and**pressurethresholdvalue**commonsametrigger**crosslayermigrationshift**. 

4. **performancebaselinevaluevalue**: with **Wish-prob / Distance-score / Fetch-distance / Random** etc.is"nonscorelayer"baseline, againaddonour **MCRPS/D/K+DRL** methodplan, formbecome"have/noscorelayerandqueueization"systemlevelComparison. 

---

## 📊 Experimental Results and Performance

* **baselineComparison**: andtransmitsystemenablesendequationmethod (Wish-prob, Distance-score, Fetch-distance, Random)Comparison, RLORMinTIS/CR/TNGOetc.Metricssignificantlysuperiorexceed
* **keyperformance**: "accordingbecomepowerratesorting"strategymostsuperior, phaseratiorandomsortinginconflictrateanddelayerrormethodaspecthavesignificantlychangeimprove
* **System Scale**: 366ten thousandexchangemutualsamplebooklargescaleverification, eachdecisionwhenmomentprocessing20nameriderinetc.scalescenario
* **actualwhenproperty**: eachreturncombineT=40stepsmovestatedecision, suitableshouldinlinerecommendrequiresrequest

## 🔄 Technical Adaptability to Our System

### Adaptability Scores

1. **movestateschedulinginnovation**: **7/10** (LSTMcodecode-solutioncodeforwardorderizationdecisionmechanismcanreferencetolayerinnerUAVscheduling)
2. **conflictprocessingcapability**: **8/10** (multiplemainbodyconflictsuppressideacanextensiontoairspaceConflict Avoidance)
3. **actualwhenperformance**: **6/10** (T=40stepsdecisionperiod, requiresneedaddspeedtohaosecondlevel)
4. **multi-objectiveoptimization**: **7/10** (multi-objectiveweightedframeworkunitscanextensiontoour7dimensionalrewardstructure)
5. **statespacedesign**: **6/10** (richfeatureworkprocesscanreference, butrequiresneedqueueizationchangecreate)

### Technical Reference Value

1. **LSTMsequenceizationdecision**: TreatsforwardorderrecommendmechanismshouldusestolayerinnerUAVtask allocation
2. **conflictsuppressmechanism**: extensionordersingleConflict Avoidancetoairspaceresourcecompetemanagement
3. **multi-objectiverewarddesign**: referenceitsconflict-efficiency-pathpathtradeoffidea
4. **movestaterecommendstrategy**: changecreateismovestatelayerbetweentransfertriggermechanism

---

**shouldusesinnovationdegree (phaseforUAVstudyresearch)**: **6/10** (thispaperinOFDTreats**multiplemainbodyconflict+movestaterecommend+multi-objectivereward**and**LSTMforwardorderdecision**systemizationimplementplaceandusestrueactuallargenumberdataevaluatetest, butnottouchandverticalairspaceandcapacity/queueingprocess). 
**oursuperiorpotentialcertainrecognize**: **significantlyimprovement** (in"verticalscorelayer+capacityallocationplacement+queueingization+pressuretransfer"dimensionaldegreehasbookqualitypoordifferenceandchangestrongcanextensionproperty). 

---

**Analysis Completion Date**: 2025-01-28 
**Analysis Quality**: Detailed analysis withmovestaterecommendmechanismandconflictprocessingmethod 
**Recommended Use**: asmovestateschedulingshouldusesreference, referenceLSTMsequenceizationdecisionandconflictsuppressmechanism
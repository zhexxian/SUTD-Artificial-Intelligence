;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Colour Spray
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain painting)
  (:requirements :strips)
  (:predicates (on ?x ?y)
	       (on-table ?x)
	       (clear ?x)
	       (arm-empty)
	       (holding ?x)
	       (color-of ?x ?color) 
	       (type-box ?x)
	       (type-color ?x) 
	       )
  (:action pick-up
	     :parameters (?ob1)
	     :precondition (and (type-box ?ob1) (clear ?ob1) (on-table ?ob1) (arm-empty))
	     :effect
	     (and (not (on-table ?ob1))
		   (not (clear ?ob1))
		   (not (arm-empty))
		   (holding ?ob1)))
  (:action put-down
	     :parameters (?ob)
	     :precondition (and (type-box ?ob) (holding ?ob))
	     :effect
	     (and (not (holding ?ob))
		   (clear ?ob)
		   (arm-empty)
		   (on-table ?ob)))
  (:action stack
	     :parameters (?sob ?sunderob)
	     :precondition (and (holding ?sob) (clear ?sunderob) (type-box ?sob) (type-box ?sunderob))
	     :effect
	     (and (not (holding ?sob))
		   (not (clear ?sunderob))
		   (clear ?sob)
		   (arm-empty)
		   (on ?sob ?sunderob)))
  (:action unstack
	     :parameters (?sob ?sunderob)
	     :precondition (and (on ?sob ?sunderob) (clear ?sob) (arm-empty) (type-box ?sob) (type-box ?sunderob))
	     :effect
	     (and (holding ?sob)
		   (clear ?sunderob)
		   (not (clear ?sob))
		   (not (arm-empty))
		   (not (on ?sob ?sunderob))))
  (:action spray
	     :parameters (?sspray ?sob ?colorspray)
	     :precondition (and (on-table ?sob) (clear ?sob) (holding ?sspray) (type-box ?sspray) (type-box ?sob) (color-of ?sspray ?colorspray) (type-color ?colorspray))
	     :effect
	     (color-of ?sob ?colorspray)))


import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from elastica import *
from elastica import (
    AnalyticalLinearDamper,
    BaseSystemCollection,
    CallBackBaseClass,
    CallBacks,
    Constraints,
    Contact,
    CosseratRod,
    Damping,
    Forcing,
    GravityForces,
    OneEndFixedBC,
    PositionVerlet,
    RodSelfContact,
    defaultdict,
)
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from muscles import ApplyMuscleGroups, MuscleGroup
from tendon import Tendon1, Tendon2

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class ArmSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping, Contact
):
    pass


class BaseArmEnvElastica:
    def __init__(
        self,
        *args,
        auto_reset: bool = True,
        seed: int | None = None,
        **kwargs,
    ):

        self.arm_stepper = PositionVerlet()
        self.COLLECT_DATA_FOR_PROCESSING = kwargs.get(
            "COLLECT_DATA_FOR_PROCESSING", False
        )
        self.time_tracker = np.float64(0.0)

        # ========== PHYSICAL PARAMETERS FOR THE ARM ==========

        self.num_elems = kwargs.get("num_elems", 100)
        self.poisson_ratio = kwargs.get("poisson_ratio", 0.5)
        self.radius = kwargs.get("radius", 0.03)  # m
        self.length = kwargs.get("length", 0.6)  # m
        self.dt = kwargs.get("dt", 1e-4)  # second
        # self.damping_nu = kwargs.get("damping_nu", 0.04)
        self.damping_nu = kwargs.get("damping_nu", 0.0048)
        self.density = kwargs.get("density", 1000)  # kg/m^3
        self.youngs_modulus = kwargs.get("youngs_modulus", 1e6)

        self.direction = kwargs.get("direction", np.array([0.0, 0.0, 1.0]))
        # self.direction = kwargs.get("direction", np.array([1.0, 0.0, 0.0]))
        self.normal = kwargs.get("normal", np.array([0.0, 1.0, 0.0]))
        self.start = kwargs.get("start", np.array([0.0, 0.0, 0.0]))

        # ========== RENDER PARAMETERS =========================
        self.step_skip = kwargs.get("step_skip", 1)
        self.plot_suffix = kwargs.get("plot_name", "test_elastica")
        self.plot_dir = kwargs.get(
            "plot_dir", os.path.join(os.getcwd(), "plots", "arm", "video")
        )
        self.plot_name = os.path.join(self.plot_dir, self.plot_suffix)
        self.target_position = kwargs.get("target_position")

        self.sim_duration = kwargs.get("sim_duration", 10)  # second

        self.sim_step = int(
            round(self.sim_duration / self.dt)
        )  # total simulation steps

    def get_state(self):
        # TODO: how to calculate the acceleration is a problem.
        # return self.arm_simulator.position_collection.position_collection
        return 0

    def reset_arm(self):
        # ========== CREATE THE ARM ============================

        self.arm_sim = ArmSimulator()

        radius = np.full(self.num_elems, self.radius)

        self.arm = CosseratRod.straight_rod(
            n_elements=self.num_elems,
            start=self.start,
            direction=self.direction,
            normal=self.normal,
            base_length=self.length,
            base_radius=radius,
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.youngs_modulus / (2 * (1 + self.poisson_ratio)),
        )
        self.arm_sim.append(self.arm)

        class ArmCallBack(CallBackBaseClass):
            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["velocity"].append(
                        system.velocity_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["mass"].append(system.mass.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    self.callback_params["avg_velocity"].append(
                        system.compute_velocity_center_of_mass()
                    )
                    self.callback_params["director"].append(
                        system.director_collection[..., -1].copy()
                    )
                    self.callback_params["director_all"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["omega"].append(system.omega_collection.copy())
                    self.callback_params["acceleration"].append(
                        system.acceleration_collection.copy()
                    )
                    return

        if self.COLLECT_DATA_FOR_PROCESSING:
            self.arm_post_processing = defaultdict(list)
            self.arm_sim.collect_diagnostics(self.arm).using(
                ArmCallBack,
                step_skip=self.step_skip,
                callback_params=self.arm_post_processing,
            )

        # ========== ADD DAMPING ===============================

        self.arm_sim.dampen(self.arm).using(
            AnalyticalLinearDamper,
            damping_constant=self.damping_nu,
            time_step=self.dt,
        )

        # ========== ADD SELF CONTACT ==========================
        self.arm_sim.detect_contact_between(self.arm, self.arm).using(
            RodSelfContact,
            k=1e4,
            nu=10,
        )

        # ========== ADD GRAVITY ==============================
        # note: only for test
        self.arm_sim.add_forcing_to(self.arm).using(
            GravityForces,
            acc_gravity=np.array([-9.81 * 0, 0, -9.81 * 1]),
        )

        # ========== ADD TENDONS ===============================

        def add_tendons(radius_base, arm):

            tendon_groups: list[MuscleGroup] = []

            tendon_ratio_radius = 0.0005 / radius_base
            shearable_rod_area = np.pi * arm.radius**2
            rest_tendon_area = shearable_rod_area * (tendon_ratio_radius**2)
            max_tendon_stress = 1e5

            tendon_groups.append(
                MuscleGroup(
                    muscles=[
                        Tendon1(
                            rest_muscle_area=rest_tendon_area,
                            max_muscle_stress=max_tendon_stress,
                            radius_list=arm.radius,
                        )
                    ],
                    type_name="tendon1",
                )
            )

            tendon_groups.append(
                MuscleGroup(
                    muscles=[
                        Tendon2(
                            rest_muscle_area=rest_tendon_area,
                            max_muscle_stress=max_tendon_stress,
                            radius_list=arm.radius,
                        )
                    ],
                    type_name="tendon2",
                )
            )

            # tendon_groups.append(
            #     MuscleGroup(
            #         muscles=[
            #             Tendon3(
            #                 rest_muscle_area=rest_tendon_area,
            #                 max_muscle_stress=max_tendon_stress,
            #                 radius_list=arm.radius,
            #             )
            #         ],
            #         type_name="tendon3",
            #     )
            # )

            # tendon_groups.append(
            #     MuscleGroup(
            #         muscles=[
            #             Tendon4(
            #                 rest_muscle_area=rest_tendon_area,
            #                 max_muscle_stress=max_tendon_stress,
            #                 radius_list=arm.radius,
            #             )
            #         ],
            #         type_name="tendon4",
            #     )
            # )

            for tendon_group in tendon_groups:
                tendon_group.set_current_length_as_rest_length(arm)

            return tendon_groups

        self.tendon_groups = add_tendons(self.radius, self.arm)
        self.tendon_post_processing_list = [
            defaultdict(list) for _ in self.tendon_groups
        ]

        self.arm_sim.add_forcing_to(self.arm).using(
            ApplyMuscleGroups,
            muscle_groups=self.tendon_groups,
            step_skip=self.step_skip,
            COLLECT_DATA_FOR_POSTPROCESSING=self.COLLECT_DATA_FOR_PROCESSING,
            callback_params_list=self.tendon_post_processing_list,
        )

        self.arm_sim.constrain(self.arm).using(
            OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

    def reset_finalize(self):
        self.arm_sim.finalize()
        self.arm_do_step, self.arm_stages_and_updates = extend_stepper_interface(
            self.arm_stepper, self.arm_sim
        )
        self.time_tracker = np.float64(0.0)
        return self.get_state()

    def reset(self):
        self.reset_arm()
        _ = self.reset_finalize()

    def save_data(
        self,
    ):
        os.makedirs(os.path.split(self.plot_name)[0], exist_ok=True)
        import pickle

        filename = f"{self.plot_name}_ROD.dat"
        file = open(filename, "wb")
        pickle.dump(self.arm_post_processing, file)
        file.close()
        print(f"ROD Data is saved to {filename}")

        filename = f"{self.plot_name}_TENDONS.dat"
        file = open(filename, "wb")
        pickle.dump(self.tendon_post_processing_list, file)
        file.close()
        print(f"TENDON Data is saved to {filename}")

    def simulate(self, action):
        action = np.array(action[::], dtype=np.float32)

        arm_tip_position = []
        time = []
        print("We are here! and begin the simulation")
        for _ in range(self.sim_step):
            # arm.position_collection shape: [3, N]
            # todo: double check
            arm_tip_position.append(self.arm.position_collection[..., -1].copy())
            time.append(self.time_tracker)

            for tendon_group, force in zip(self.tendon_groups, action):
                tendon_group.apply_activation(force)

            self.time_tracker = self.arm_do_step(
                self.arm_stepper,
                self.arm_stages_and_updates,
                self.arm_sim,
                self.time_tracker,
                self.dt,
            )

        # (5) crash/NaN handling (match soromox style)
        invalid_arm = _isnan_check(self.arm.position_collection) or _isnan_check(
            self.arm.velocity_collection
        )

        if invalid_arm:
            print("BE carefully !!! Something is NaN output!!!")
            obs = -np.inf
        else:
            arm_tip_position = np.array(arm_tip_position)
            obs = arm_tip_position
            time = np.array(time)

        return obs, time


arm = BaseArmEnvElastica(sim_duration=3)
arm.reset()

input = np.array([1, 1])

start = time.perf_counter()

obs, time_sim = arm.simulate(input)

elapsed = time.perf_counter() - start
print(f"Elapsed time: {elapsed:.6f} seconds")

print(obs.shape)
print(time_sim.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.plot(time_sim, obs[:, 0])
plt.ylabel("x [m]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_sim, obs[:, 1])
plt.ylabel("y [m]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_sim, obs[:, 2])
plt.ylabel("z [m]")
plt.xlabel("time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()

x = obs[:, 0]
y = obs[:, 1]
z = obs[:, 2]

df = pd.DataFrame({"time": time_sim, "x": x, "y": y, "z": z})

# Save to Excel

DATA_DIR.mkdir(parents=True, exist_ok=True)
file_path = DATA_DIR / "end_effector_position_tendon_driven_pyelastica.xlsx"
df.to_excel(file_path, index=False)

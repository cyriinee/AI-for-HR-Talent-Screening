def normalize_skills(skills):
    return sorted({s.strip().lower() for s in skills if s and s.strip()})

def compute_skill_plan(candidate_skills, job_skills):
    cand = set(normalize_skills(candidate_skills))
    job = set(normalize_skills(job_skills))
    gaps = sorted(job - cand)
    overlap = sorted(job & cand)
    return {
        "gaps": gaps,
        "overlap": overlap,
        "job": sorted(job),
        "candidate": sorted(cand),
    }

if __name__ == "__main__":
    cand = ["python", "react", "sql basics"]
    job = ["python", "django", "rest apis", "async javascript", "sql"]
    print(compute_skill_plan(cand, job))
